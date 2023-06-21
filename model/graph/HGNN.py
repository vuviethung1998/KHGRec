import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import time
from os.path import abspath
import wandb 

from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise, next_batch_pairwise_kg
import torch
import torch.nn as nn 
import torch.nn.functional as F
from scipy.sparse import coo_matrix
import os
import numpy as np 
import time 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util.loss_torch import bpr_loss, l2_reg_loss, EmbLoss, contrastLoss
from util.init import *
from base.torch_interface import TorchGraphInterface

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HGNN(GraphRecommender):
    def __init__(self, conf, training_set, test_set, knowledge_set, **kwargs):
        GraphRecommender.__init__(self, conf, training_set, test_set, knowledge_set, **kwargs)
        # config = OptionConf(self.config['HGNN'])

        self.reg_loss = EmbLoss() 
        self.model = HGNNModel(self.config, self.data )

        self._parse_config(self.config)
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), weight_decay=1e-3, lr=self.lRate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=self.lr_decay,patience=7)

    def _parse_config(self, config):
        self.lRate = float(config['learnRate'])
        self.lr_decay = float(config['learnRateDecay'])
        self.maxEpoch = int(config['num.max.epoch'])
        self.batchSize = int(config['batch_size'])
        self.reg = float(config['reg.lambda'])
        # self.embeddingSize = wandb.config.input_dim
        self.hyperDim = int(config['hyper.size'])
        # self.hyperDim = wandb.config.hyper_dim  
        self.dropRate = float(config['dropout'])
        self.negSlove = float(config['leaky'])
        self.nLayers = int(config['gnn_layer'])

    def train(self):
        model = self.model 
        train_losses = []

        for ep in range(self.maxEpoch):
            train_losses = []
            rec_losses = []
            reg_losses = []

            for n, batch in enumerate(next_batch_pairwise_kg(self.data, self.batchSize)):
                user_idx, pos_idx, neg_idx, entity_idx = batch
                model.train()
                # s_model = time.time()
                rec_user_emb, rec_item_emb = model()

                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                loss_rec, loss_reg = self.calculate_loss(user_emb, pos_item_emb, neg_item_emb, self.batchSize)

                batch_loss = loss_rec + loss_reg 
                
                train_losses.append(batch_loss.item())
                rec_losses.append(loss_rec.item())
                reg_losses.append(loss_reg.item())
                
                self.optimizer.zero_grad()
                batch_loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 4)
                self.optimizer.step()
            
            train_loss = np.mean(train_losses)
            rec_loss = np.mean(rec_losses)
            reg_loss = np.mean(reg_losses)

            train_losses.append([ep, train_loss])
            rec_losses.append([ep,rec_loss])
            reg_losses.append([ep, reg_loss])

            model.eval()
            with torch.no_grad():
                self.user_emb, self.item_emb = model()
            s_eval = time.time()
            self.fast_evaluation(ep)
            e_eval = time.time()
            print("Eval time: %f s" % (e_eval - s_eval))
        self.save_loss(train_losses, rec_losses, reg_losses)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model()
        self.save_model(self.model)

    def predict(self, u):
        user_id  = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[user_id], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

    def calculate_loss(self, anchor_emb, pos_emb, neg_emb, batch_size):
        calc_reg_loss = EmbLoss()
        rec_loss = bpr_loss(anchor_emb, pos_emb, neg_emb)
        reg_loss = self.reg * calc_reg_loss(anchor_emb, pos_emb, neg_emb) / batch_size
        return rec_loss, reg_loss

class HGNNModel(nn.Module):
    def __init__(self, config, data):
        super(HGNNModel, self).__init__()
        self.data = data
        self._parse_args(config)
        
        H = self.data.interaction_mat
        H_T = self.data.inv_interaction_mat
        G_e = self.data.hyperedge_adj
        G_v = self.data.hypervertex_adj
        
        self.H = TorchGraphInterface.convert_sparse_mat_to_tensor(H).to_dense().to(device)
        self.H_T = TorchGraphInterface.convert_sparse_mat_to_tensor(H_T).to_dense().to(device)
        self.G_e = TorchGraphInterface.convert_sparse_mat_to_tensor(G_e).to_dense().to(device)
        self.G_v = TorchGraphInterface.convert_sparse_mat_to_tensor(G_v).to_dense().to(device)
        self.hgnn_u = HGNNLayer(self.H, self.G_e, self.G_v, self.input_dim, self.hyper_dim, self.data.user_num, self.data.item_num, self.p, self.drop_rate)
        self.hgnn_i = HGNNLayer(self.H_T, self.G_v, self.G_e, self.input_dim, self.hyper_dim, self.data.item_num, self.data.user_num, self.p, self.drop_rate)

        # init embedding
        self.user_embedding = nn.Embedding(self.data.user_num, self.input_dim)
        self.item_embedding = nn.Embedding(self.data.item_num, self.input_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
    def _parse_args(self, config):
        self.input_dim = int(config['embedding.size'])
        self.hyper_dim = int(config['hyper.size'])
        self.drop_rate = float(config['dropout'])
        self.p = float(config['leaky'])

    def forward(self):        
        u_embed = self.user_embedding.weight.cuda()
        i_embed = self.item_embedding.weight.cuda()
        user_embed = self.hgnn_u(u_embed)
        item_embed = self.hgnn_i(i_embed)
        return user_embed, item_embed 

class HGNNConv(nn.Module):
    def __init__(self, H, G_e, G_v, input_dim, output_dim, n_vertices, n_edges, p, drop_rate=0.1, skip_connection=False):
        '''The hypergraph convolutional operator from the `"Hypergraph Convolution
            and Hypergraph Attention" <https://arxiv.org/abs/1901.08150>`_ paper
                X(l+1) = σ( D_v^(−1/2)HW_eD_e^(-1)H.T D_v^(−1/2) X(l) W ) 
            where:
                X(l): input vertex feature matrix (n,F)
                H: the hypergraph incidence matrix (n,m)
                D_e: diagonal hyperedge degree matrix (m,m)
                D_v: diagonal vertex degree matrix (n,n)
                W_e: diagonal hyperedge weight matrix 
                W: learnable parameters.
            Args:
                ``input_dim`` (``int``): :math:`C_{in}` is the number of input channels.
                ``output_dim`` (int): :math:`C_{out}` is the number of output channels.
                ``bias`` (``bool``): If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
                ``p`` (``float``): If set to a positive number, the layer will use dropout. Defaults to ``0.5``.
                ``is_last`` (``bool``): If set to ``True``, the layer will not apply the final activation and dropout functions. Defaults to ``False``.
        '''
        super(HGNNConv, self).__init__()
        self.skip_connection = skip_connection
        self.W_e = Parameter(torch.eye(n_edges), requires_grad=True) # (m,m)
        self.W = Parameter(torch.rand(input_dim,output_dim),requires_grad=True) # (in, out)
        self.bias = Parameter(torch.rand(output_dim),requires_grad=True) 
        
        self.non_linear = nn.LeakyReLU(p)
        self.drop_out = nn.Dropout(drop_rate)
        self.apply(xavier_uniform_initialization)
        
        # calc 
        I_e = torch.eye(G_e.shape[0]).to(device)
        G_e_hat = G_e + I_e
        D_e_inv = torch.inverse(torch.diag(torch.sum(G_e_hat, axis=0))) # (m,m)
        
        I_v = torch.eye(G_v.shape[0]).to(device)
        G_v_hat = G_v + I_v
        D_v_diag_inv = torch.rsqrt(torch.diag(torch.sum(G_v_hat, axis=0))) # (n,n)
        D_v_diag_inv[torch.isinf(D_v_diag_inv)] = 0 # inf -> 0
        
        self.D_v_diag_inv = D_v_diag_inv
        self.D_e_inv = D_e_inv
        self.H = H
        
    def forward(self, x) -> torch.Tensor:
        DvH = torch.mm(self.D_v_diag_inv, self.H)
        DvHWe = torch.mm(DvH, self.W_e)
        DvHWeDe = torch.mm(DvHWe, self.D_e_inv)
        DvHWeDeHt = torch.mm(DvHWeDe, self.H.T)
        DvHWeDeHtDv = torch.mm(DvHWeDeHt, self.D_v_diag_inv)
        
        out = torch.mm(torch.mm(DvHWeDeHtDv, x), self.W) + self.bias
        if self.skip_connection:
            out += x 
        return out

class HGNNLayer(nn.Module):
    def __init__(self, H, G_e, G_v, input_dim, hyper_dim, n_vertices, n_edges, p=0.1, drop_rate=0.1, skip_connection=False):
        super(HGNNLayer, self).__init__()
#         self.skip = skip_connection
        self.fc = nn.Linear(input_dim, hyper_dim, bias=False)
        self.non_linear = nn.LeakyReLU(p)
        self.drop_out = nn.Dropout(drop_rate)
        self.hgnn1 = HGNNConv(H, G_e, G_v, hyper_dim, hyper_dim, n_vertices, n_edges, p, drop_rate, skip_connection = skip_connection)
        self.hgnn2 = HGNNConv(H, G_e, G_v, hyper_dim, hyper_dim, n_vertices, n_edges, p, drop_rate, skip_connection = skip_connection)

    def forward(self, x):
        '''
        Apply skip connection
        '''
        lat1 = self.drop_out(self.non_linear(self.fc(x)))
        out = self.drop_out(self.non_linear(self.hgnn1(lat1)))
        lat2 = self.drop_out(self.non_linear(self.hgnn1(lat1)))
        out = self.drop_out(self.hgnn2(lat2))
        return out
