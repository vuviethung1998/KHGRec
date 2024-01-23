import torch
import torch.nn as nn
import numpy as np 
import pandas as pd 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss,l2_reg_loss
from util.evaluation import early_stopping

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

        self.model = HGCN_Encoder(self.data, self.emb_size, self.hyper_size, self.n_layers, self.leaky, self.drop_rate, self.device)

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
                rec_user_emb, rec_item_emb = model(keep_rate=1-self.drop_rate)
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
    def __init__(self, data, emb_size, hyper_size, n_layers, leaky, drop_rate, device):
        super(HGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.hyper_size = hyper_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
        self.relu = nn.ReLU()
        self.act = nn.LeakyReLU(leaky)
        self.dropout = nn.Dropout(drop_rate)
        self.edgeDropper = SpAdjDropEdge()

        self.residuals = torch.nn.ModuleList()

        self.hgnn_layers = []
        self.ugformer_layers = []

        for i in range(self.layers):
            encoder_layers = TransformerEncoderLayer(d_model=hyper_size, nhead=2, dim_feedforward=32, dropout=drop_rate) # Default batch_first=False (seq, batch, feature)
            enc_norm = nn.LayerNorm(hyper_size)
            self.ugformer_layers.append(TransformerEncoder(encoder_layers, 1, norm=enc_norm).to(device))

            if i ==0:
                self.hgnn_layers.append(HGCNConv(leaky=leaky))
            else:
                self.hgnn_layers.append(HGCNConv(leaky=leaky))

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.n_users, self.hyper_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.n_items, self.hyper_size))),
        })
        return embedding_dict

    def forward(self, keep_rate=1):
        self.user_emb =  self.embedding_dict['user_emb'] 
        self.item_emb = self.embedding_dict['item_emb']  
        ego_embeddings = torch.cat([self.user_emb, self.item_emb], 0)
        sparse_norm_adj = self.edgeDropper(self.sparse_norm_adj, keep_rate)

        res = ego_embeddings
        all_embeddings = []
        for k in range(self.layers):
            # self-attention over all nodes
            input_Tr = torch.unsqueeze(ego_embeddings, 1)  #[seq_length, batch_size=1, dim] for pytorch transformer
            input_Tr = self.ugformer_layers[k](input_Tr)
            ego_embeddings = torch.squeeze(input_Tr, 1)
            if k != self.layers - 1: 
                ego_embeddings = self.hgnn_layers[k](sparse_norm_adj, ego_embeddings)
            else:
                ego_embeddings = self.hgnn_layers[k](sparse_norm_adj, ego_embeddings, act=False) 
            all_embeddings += [ego_embeddings]
        
        all_embeddings[-1] += res
        user_all_embeddings = all_embeddings[-1][:self.data.n_users]
        item_all_embeddings = all_embeddings[-1][self.data.n_users:]
        return user_all_embeddings, item_all_embeddings

class HGCNConv(nn.Module):
    def __init__(self, leaky):
        super(HGCNConv, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=leaky)

    def forward(self, adj, embs, act=True):
        if act:
            return self.act(torch.sparse.mm(adj, torch.sparse.mm(adj.t(), embs)))
        else:
            return torch.sparse.mm(adj, torch.sparse.mm(adj.t(), embs))

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
    
