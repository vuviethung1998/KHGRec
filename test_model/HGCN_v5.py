import torch
import torch.nn as nn
import numpy as np 
import pandas as pd 

from torch.optim.lr_scheduler import ReduceLROnPlateau

from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss,l2_reg_loss
from util.evaluation import early_stopping
from data.augmentor import drop_edges
from model.layers.HypergraphConv import HypergraphConv
# from torch_geometric.nn import HypergraphConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class HGCN(GraphRecommender):
    def __init__(self, conf, training_set, test_set, knowledge_set, **kwargs):
        super(HGCN, self).__init__(conf, training_set, test_set, knowledge_set, **kwargs)
        self.device = torch.device(f"cuda:{kwargs['gpu_id']}" if torch.cuda.is_available() else 'cpu')
        self.n_layers = int(kwargs['n_layers'])

        self.early_stopping_steps = int(kwargs['early_stopping_steps']) 
        self.weight_decay = float(kwargs['weight_decay'])
        self.emb_size =  int(kwargs['input_dim'])
        self.hyper_size =  int(kwargs['hyper_dim'])
        self.leaky = float(kwargs['p'])
        self.drop_rate = float(kwargs['drop_rate'])
        self.reg = float(kwargs['reg'])
        self.nheads = int(kwargs['nheads'])
        self.model = HGCN_Encoder(self.data, self.emb_size, self.hyper_size, self.n_layers, self.leaky, self.drop_rate, self.nheads)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=self.lr_decay, patience=10)

        recall_list = []
        lst_train_losses = []
        lst_rec_losses = []
        lst_reg_losses = []
        lst_performances = []

        for epoch in range(self.maxEpoch):
            train_losses = []
            rec_losses = []
            reg_losses = []
            
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) 
                reg_loss = l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb) /self.batch_size
                batch_loss = rec_loss + reg_loss

                train_losses.append(batch_loss.item())
                rec_losses.append(rec_loss.item())
                reg_losses.append(reg_loss.item())

                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
                
            train_loss = np.mean(train_losses)
            rec_loss = np.mean(rec_losses)
            reg_loss = np.mean(reg_losses)

            scheduler.step(train_loss)

            lst_train_losses.append([epoch, train_loss])
            lst_rec_losses.append([epoch,rec_loss])
            lst_reg_losses.append([epoch, reg_loss])
            
            with torch.no_grad():
                self.user_emb, self.item_emb = model()
            # if epoch % 5 == 0:
            measure, data_ep = self.fast_evaluation(epoch)
            lst_performances.append(data_ep)

            cur_recall =  float(measure[2].split(':')[1])
            recall_list.append(cur_recall)
            best_recall, should_stop = early_stopping(recall_list, self.early_stopping_steps)
            if should_stop:
                break 
        
        print(lst_train_losses)
        self.save_loss(lst_train_losses, lst_rec_losses, lst_reg_losses)
        self.save_perfomance_training(lst_performances)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()
            print("Saving")
            self.save_model(self.model)     

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class HGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, hyper_size, n_layers, leaky, drop_rate, nheads):
        super(HGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.hyper_size = hyper_size
        self.layers = n_layers
        self.drop_rate = drop_rate

        self.edge_index = data.edge_index.cuda()
        self.edge_index_t = data.edge_index_t.cuda()
        self.embedding_dict = self._init_model()
        self.relu = nn.ReLU()
        self.act = nn.LeakyReLU(leaky)
        self.dropout = nn.Dropout(drop_rate)

        self.use_drop_edge = False
        self.hgnn_layer_u = SelfAwareHGCNConv(leaky=leaky, dropout=drop_rate, n_layers=n_layers, nheads=nheads, input_dim=self.latent_size, hidden_dim=64, hyper_dim=self.hyper_size, bias=True).cuda()
        self.hgnn_layer_i = SelfAwareHGCNConv(leaky=leaky, dropout=drop_rate, n_layers=n_layers, nheads=nheads, input_dim=self.latent_size, hidden_dim=64, hyper_dim=self.hyper_size, bias=True).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.n_users, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.n_items, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        self.user_emb =  self.embedding_dict['user_emb']
        self.item_emb = self.embedding_dict['item_emb']

        if self.use_drop_edge:
            self.edge_index = drop_edges(self.edge_index, aug_ratio=self.drop_rate)
        ego_embeddings = torch.cat([self.user_emb, self.item_emb], 0)
        hyperLat1 = self.hgnn_layer_u(ego_embeddings, self.edge_index, ego_embeddings)
        hyperLat2 = self.hgnn_layer_i(ego_embeddings, self.edge_index_t, ego_embeddings)
        hyperULat = hyperLat1[:self.data.n_users]
        hyperILat = hyperLat2[self.data.n_users:]
        return hyperULat, hyperILat

class HGCNConv(nn.Module):
    def __init__(self, leaky, dropout, input_dim, hyper_dim):
        super(HGCNConv, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=leaky)
        self.dropout = nn.Dropout(dropout)
        self.W = nn.Linear(input_dim, hyper_dim)
        
    def forward(self, adj, embs, act=True):
        adj_ = adj + torch.eye(adj.shape)
        D_v = torch.sum(adj_,dim=1)
        D_e = torch.sum(adj_,dim=0)
        D_v_invsqrt = torch.diag(torch.pow(D_v, -0.5))
        D_e_inv = torch.diag(torch.inverse(D_e))
        
        n_edges = D_e.shape[0]
        B = torch.eye(n_edges)
        L  = D_v_invsqrt @ adj_ @ B @ D_e_inv @ adj_.T @ D_v_invsqrt
        if act:
            return self.act(torch.sparse.mm(L, self.W(self.dropout(embs))))
        else:    
            return torch.sparse.mm(L, self.W(self.dropout(embs)))
        # if act:
        #     return self.act(torch.sparse.mm(adj, torch.sparse.mm(adj.t(), embs))) + embs
        # else:
        #     return torch.sparse.mm(adj, torch.sparse.mm(adj.t(), embs)) + embs

class SelfAwareHGCNConv(nn.Module):
    def __init__(self, leaky, dropout, n_layers, nheads, input_dim, hidden_dim, hyper_dim, att_mode='node', bias=True):
        super(SelfAwareHGCNConv, self).__init__()

        self.input_dim = input_dim
        self.hyper_dim = hyper_dim        
        self.act = nn.LeakyReLU(negative_slope=leaky)

        self.relu = nn.ReLU()
        self.leaky = leaky 
        self.dropout = dropout 
        self.n_layers = n_layers
        self.res_fc = nn.Linear(input_dim, hyper_dim).cuda()
        self.convs = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        self.residuals = torch.nn.ModuleList()
        self.hyperedge_fc = torch.nn.ModuleList()
        self.ugformer_layers = torch.nn.ModuleList()

        for i in range(n_layers):
            first_channels = input_dim if i == 0 else hidden_dim
            second_channels = hyper_dim if i == n_layers - 1 else hidden_dim
            
            encoder_layers = TransformerEncoderLayer(d_model=second_channels, nhead=nheads, dim_feedforward=hyper_dim, dropout=dropout, norm_first=True) # Default batch_first=False (seq, batch, feature)
            self.ugformer_layers.append(TransformerEncoder(encoder_layers, 4))
            
            self.convs.append(HypergraphConv(first_channels, second_channels, use_attention=False, heads=nheads, attention_mode=att_mode,\
                                            concat=False, negative_slope=leaky, dropout=dropout, bias=bias))
            self.lns.append(torch.nn.LayerNorm(second_channels))
            self.residuals.append(nn.Linear(input_dim, second_channels).cuda())
            self.hyperedge_fc.append(nn.Linear(input_dim, first_channels).cuda())

    def forward(self, inp, adj, hyperedge_attr=None):
        embs = inp
        for i, conv in enumerate(self.convs):
            residual = self.residuals[i](inp)
            
            if i != self.n_layers - 1:
                embs = self.act(self.lns[i](conv(embs, adj))) + residual
            else:
                embs = self.lns[i](conv(embs, adj)) + residual
                
            # self-attention over all nodes
            input_Tr = torch.unsqueeze(embs, 1)  #[seq_length, batch_size=1, dim] for pytorch transformer
            input_Tr = self.ugformer_layers[i](input_Tr)
            embs = torch.squeeze(input_Tr, 1)

        return embs 

class SpAdjDropEdge(nn.Module):
	def __init__(self):
		super(SpAdjDropEdge, self).__init__()

	def forward(self, adj, keepRate):
		if keepRate == 1.0:
			return adj
		vals = adj._values()
		idxs = adj._indices()
		edgeNum = vals.size()
		mask = ((torch.rand(edgeNum) + keepRate).floor()).type(torch.bool)
		newVals = vals[mask] / keepRate
		newIdxs = idxs[:, mask]
		return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)
    

