import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os 
import numpy as np 

import time
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from util.loss_torch import bpr_loss, l2_reg_loss, EmbLoss, contrastLoss
from util.init import *
from base.torch_interface import TorchGraphInterface
from torch.optim.lr_scheduler import ReduceLROnPlateau

from util.evaluation import early_stopping

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HCCF(GraphRecommender):
    def __init__(self, conf, training_set, test_set, knowledge_set, **kwargs):
        GraphRecommender.__init__(self, conf, training_set, test_set, knowledge_set, **kwargs)
        self.reg_loss = EmbLoss() 
        self.model = HCCFEncoder(kwargs, self.data )
        self._parse_config(self.config, kwargs)
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lRate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=self.lr_decay, patience=5)
        
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
        self.ss_rate = float(kwargs['cl_rate'])
        
        self.hyperDim = int(config['hyper.size'])
        self.dropRate = float(config['dropout'])
        self.negSlove = float(config['leaky'])
        self.temp = float(config['temp'])
        self.seed = int(kwargs['seed'])
        self.early_stopping_steps = int(kwargs['early_stopping_steps'])

    def calcLosses(self, ancs, poss, negs, gcnEmbedsLst, hyperEmbedsLst, reg):
        bprLoss = bpr_loss(ancs, poss, negs)
        sslLoss = 0
        for i in range(self.nLayers):
            embeds1 = gcnEmbedsLst[i].detach()
            embeds2 = hyperEmbedsLst[i]
            sslLoss += contrastLoss(embeds1[:self.data.n_users], embeds2[:self.data.n_users], torch.unique(ancs.long()), self.temp)\
                         + contrastLoss(embeds1[self.data.n_users:], embeds2[self.data.n_users:], torch.unique(poss.long()), self.temp)
        sslLoss *= self.ss_rate
        return bprLoss, sslLoss

    def train(self, load_pretrained=False):
        model = self.model 
        recall_list = []
        train_losses = [] 
        lst_performances = []

        for ep in range(self.maxEpoch):
            s_train = time.time()

            for n, batch in enumerate(next_batch_pairwise(self.data, self.batchSize)):
                user_idx, pos_idx, neg_idx = batch
                model.train()
                user_emb, item_emb, gcnEmbedsLst, hyperEmbedsLst = model(keep_rate=1- self.dropRate)

                anchor_emb = user_emb[user_idx]
                pos_emb = item_emb[pos_idx]
                neg_emb = item_emb[neg_idx]
                
                loss_rec, loss_ssl = self.calcLosses(anchor_emb, pos_emb, neg_emb, gcnEmbedsLst, hyperEmbedsLst, self.reg)
                batch_loss = loss_rec + loss_ssl  
                
                train_losses.append(batch_loss.item())
                
                self.optimizer.zero_grad()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 4)
                batch_loss.backward()
                self.optimizer.step()

            e_train = time.time() 
            tr_time = e_train - s_train 
            
            model.eval()
            with torch.no_grad():
                self.user_emb, self.item_emb, _, _ = model(keep_rate=1)
                s_eval = time.time()
                if ep >= 0:
                    cur_data, data_ep = self.fast_evaluation(ep, train_time=tr_time)
                    lst_performances.append(data_ep)
                    print(data_ep)
                    cur_recall =  float(cur_data[2].split(':')[1])
                    recall_list.append(cur_recall)
                    best_recall, should_stop = early_stopping(recall_list, self.early_stopping_steps)
                    if should_stop:
                        break 
                    
            train_loss = np.mean(train_losses)
            self.scheduler.step(train_loss)

            e_eval = time.time()
            print("Eval time: %f s" % (e_eval - s_eval))

        self.save_perfomance_training(lst_performances)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb, _,_ = self.model(keep_rate=1)
            print("Saving")
            self.save_model(self.model)     
            
    def predict(self, u):
        user_id  = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[user_id], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class HCCFEncoder(nn.Module):
    def __init__(self, conf, data):
        super(HCCFEncoder, self).__init__()
        self.data = data
        self._parse_config(conf)
        self.gcnlayer = GCNLayer(self.leaky)
        self.hgnnlayer = HGNNLayer(self.leaky)
        self.norm_adj = data.norm_adj

        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).to(device)
        self.embedding_dict = self._init_model()
        self.drop_out = nn.Dropout(self.drop_rate)
        self.edgeDropper = SpAdjDropEdge()

    def _parse_config(self, config):
        self.lRate = float(config['lrate'])
        self.lr_decay = float(config['lr_decay'])
        self.maxEpoch = int(config['max_epoch'])
        self.batchSize = int(config['batch_size'])
        self.reg = float(config['reg'])
        self.latent_size = int(config['embedding_size'])
        self.hyperDim = int(config['hyper_dim'])
        self.drop_rate = float(config['drop_rate'])
        self.leaky = float(config['p'])
        self.n_layers = int(config['n_layers'])
        self.n_edges = int(config['hyper_dim'])
        
    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.n_users, self.latent_size)).to(device)),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.n_items, self.latent_size)).to(device)),
            'user_w': nn.Parameter(initializer(torch.empty(self.latent_size, self.n_edges)).to(device)),
            'item_w': nn.Parameter(initializer(torch.empty(self.latent_size, self.n_edges)).to(device))
        })
        return embedding_dict
        
    def forward(self, keep_rate=0.5):
        embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        hidden = [embeddings]
        gcn_hidden = []
        hgnn_hidden = []
        hyper_uu = self.embedding_dict['user_emb'] @ self.embedding_dict['user_w']
        hyper_ii = self.embedding_dict['item_emb'] @self.embedding_dict['item_w']
        
        for i in range(self.n_layers):
            gcn_emb = self.gcnlayer(self.edgeDropper(self.sparse_norm_adj, keep_rate), hidden[-1]) 
            hyper_uemb = self.hgnnlayer(self.drop_out(hyper_uu), hidden[-1][:self.data.n_users])
            hyper_iemb = self.hgnnlayer(self.drop_out(hyper_ii), hidden[-1][self.data.n_users:])
            gcn_hidden += [gcn_emb]
            hgnn_hidden += [torch.cat([hyper_uemb, hyper_iemb], 0)]
            hidden += [gcn_emb + hgnn_hidden[-1]]
        embeddings = sum(hidden)
        user_emb = embeddings[:self.data.n_users]
        item_emb = embeddings[self.data.n_users:]
        return user_emb, item_emb, gcn_hidden, hgnn_hidden
        
class GCNLayer(nn.Module):
    def __init__(self, leaky):
        super(GCNLayer, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=leaky)
        
    def forward(self, adj, embeds):
        return (torch.sparse.mm(adj, embeds))
    
class HGNNLayer(nn.Module):
    def __init__(self, leaky):
        super(HGNNLayer, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=leaky)
    
    def forward(self, adj, embeds):
        #edge_embeds = self.act(torch.mm(adj.T, embeds))
        #hyper_embeds = self.act(torch.mm(adj, edge_embeds))
        edge_embeds = torch.mm(adj.T, embeds)
        hyper_embeds = torch.mm(adj, edge_embeds)
        return hyper_embeds
    
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
    