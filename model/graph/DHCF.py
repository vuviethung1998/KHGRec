import torch
import torch.nn as nn
import numpy as np 
import pandas as pd 
import time 

from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss,l2_reg_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util.evaluation import early_stopping

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DHCF(GraphRecommender):
    def __init__(self, conf, training_set, test_set, knowledge_set, **kwargs):
        super(DHCF, self).__init__(conf, training_set, test_set, knowledge_set, **kwargs)
        self.kwargs = kwargs
        self.model = DHCF_Encoder(self.config, self.data, kwargs)
        self._parse_config(self.config, kwargs)
        
    def _parse_config(self, config, kwargs):
        self.maxEpoch = int(kwargs['max_epoch'])
        self.batchSize = int(kwargs['batch_size'])
        
        self.lRate = float(kwargs['lrate'])
        self.lr_decay = float(kwargs['lr_decay'])
        self.maxEpoch = int(kwargs['max_epoch'])
        self.batchSize = int(kwargs['batch_size'])
        self.reg = float(kwargs['reg'])
        self.latent_size = int(kwargs['embedding_size'])
        self.hyperDim = int(kwargs['hyper_dim'])
        self.drop_rate = float(kwargs['drop_rate'])
        self.leaky = float(kwargs['p'])
        self.nLayers = int(kwargs['n_layers'])
        self.cl_rate = float(kwargs['cl_rate'])
        self.temp = float(kwargs['temp'])
        self.seed = int(kwargs['seed'])
        self.weight_decay =float(kwargs['weight_decay'])
        self.early_stopping_steps = int(kwargs['early_stopping_steps'])
        
    def train(self, load_pretrained):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=self.lr_decay, patience=5)

        lst_train_losses = []
        lst_rec_losses = []
        lst_reg_losses = []
        lst_performances = []
        recall_list = []
        
        for epoch in range(self.maxEpoch):
            train_losses = []
            rec_losses = []
            reg_losses = []
            
            s_train = time.time()

            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) 
                reg_loss =  l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb) /self.batch_size
                batch_loss = rec_loss + reg_loss

                train_losses.append(batch_loss.item())
                rec_losses.append(rec_loss.item())
                reg_losses.append(reg_loss.item())

                # Backward and optimize
                optimizer.zero_grad()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 4)
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            
            e_train = time.time() 
            tr_time = e_train - s_train 

            train_loss = np.mean(train_losses)
            rec_loss = np.mean(rec_losses)
            reg_loss = np.mean(reg_losses)

            lst_train_losses.append([epoch, train_loss])
            lst_rec_losses.append([epoch,rec_loss])
            lst_reg_losses.append([epoch, reg_loss])
            scheduler.step(train_loss)

            with torch.no_grad():
                self.user_emb, self.item_emb = model()
                cur_data, data_ep = self.fast_evaluation(epoch, train_time=tr_time)
                lst_performances.append(data_ep)
                
                cur_recall =  float(cur_data[2].split(':')[1])
                recall_list.append(cur_recall)
                best_recall, should_stop = early_stopping(recall_list, self.early_stopping_steps)
                if should_stop:
                    break 
        
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

class HGCNConv(nn.Module):
    def __init__(self, leaky):
        super(HGCNConv, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=leaky)

    def forward(self, adj, embs, act=True):
        if act:
            return self.act(torch.sparse.mm(adj, torch.sparse.mm(adj.t(), embs)))
        else:
            return torch.sparse.mm(adj, torch.sparse.mm(adj.t(), embs))

class DHCF_Encoder(nn.Module):
    def __init__(self, config, data, args):
        super(DHCF_Encoder, self).__init__()
        self.data = data
        adj = data.interaction_mat
        self.adj  = TorchGraphInterface.convert_sparse_mat_to_tensor(adj).to_dense().to(device)
        
        self._parse_args(args)
        self.embedding_dict = self._init_model()
        
        self.fc_u = nn.Linear(self.hyper_dim, self.hyper_dim)
        self.fc_i = nn.Linear(self.hyper_dim, self.hyper_dim)
        
        self.hgnn_u =  [HGCNConv(leaky=self.p) for _ in range(self.layers)]
        self.hgnn_i = [HGCNConv(leaky=self.p) for _ in range(self.layers)]
        
        self.non_linear = nn.ReLU()
        self.dropout = nn.Dropout(self.drop_rate)
        
    def _parse_args(self, args):
        self.input_dim = args['input_dim']
        self.hyper_dim = args['hyper_dim']
        self.p = args['p']
        self.drop_rate = args['drop_rate'] 
        self.layers = args['n_layers']
    
    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.n_users, self.hyper_dim)).to(device)),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.n_items, self.hyper_dim)).to(device))
        })
        return embedding_dict

    def forward(self):
        uEmbed = self.embedding_dict['user_emb']
        iEmbed = self.embedding_dict['item_emb']
        
        user_embeds = [uEmbed]
        item_embeds = [iEmbed]
        
        for idx, layer in enumerate(range(self.layers)):
            hyper_u_embed = self.hgnn_u[idx](self.adj, uEmbed)
            hyper_i_embed = self.hgnn_i[idx](self.adj.T, iEmbed)
            
            user_embeds.append(hyper_u_embed)
            item_embeds.append(hyper_i_embed)
        
        user_all_embeddings = torch.cat(user_embeds, dim=1)
        item_all_embeddings = torch.cat(item_embeds, dim=1)
        
        return user_all_embeddings, item_all_embeddings 


