import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import time
from os.path import abspath
import wandb 
from tqdm import tqdm 

from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise, next_batch_pairwise_kg, next_batch_pairwise_kg_neg, next_batch_pairwise_kg_neg_relation
import torch
import torch.nn as nn 
import torch.nn.functional as F
from scipy.sparse import coo_matrix
import os
import numpy as np 
import time 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util.loss_torch import bpr_loss, l2_reg_loss, EmbLoss, InfoNCE
from util.init import *
from base.torch_interface import TorchGraphInterface
from data.augmentor import GraphAugmentor

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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lRate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=self.lr_decay,patience=5)

    def _parse_config(self, config):
        self.model_name = config['model.name']
        self.dataset = config['dataset']

        self.lRate = float(config['learnRate'])
        self.lr_decay = float(config['learnRateDecay'])
        self.maxEpoch = int(config['num.max.epoch'])
        self.batchSize = int(config['batch_size'])
        self.reg = float(config['reg.lambda'])
        # self.embeddingSize = wandb.config.input_dim
        self.hyper_dim = int(config['hyper.size'])
        self.input_dim = int(config['input.size'])
        self.drop_rate = float(config['dropout'])
        self.p = float(config['leaky'])
        self.layers = int(config['gnn_layer'])
        self.cl_rate = float(config['cl_rate'])

    def train(self):
        model = self.model 
        final_train_losses = []
        final_rec_losses = []
        final_reg_losses = []
        final_cl_losses = []

        for ep in range(self.maxEpoch):
            train_losses = []
            rec_losses = []
            reg_losses = []
            cl_losses = []

            dropped_adj1 = model.graph_reconstruction()
            dropped_adj2 = model.graph_reconstruction()

            for  batch in tqdm(next_batch_pairwise(self.data, self.batchSize)):
                user_idx, pos_idx, neg_idx = batch
                model.train()
                # s_model = time.time()
                rec_user_emb, rec_item_emb = model()

                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss, reg_loss = self.calculate_loss(user_emb, pos_item_emb, neg_item_emb, self.batchSize)
                # kg_loss, reg_kg_loss = calculate_kg_loss(anchor_i_emb, ent_emb, ent_emb_neg, batchSize, reg)
                # ssl_loss = calculate_ssl_loss()
                cl_loss = self.cl_rate * model.cal_cl_loss([user_idx, pos_idx], dropped_adj1,dropped_adj2)

                batch_loss = rec_loss + reg_loss + cl_loss

                train_losses.append(batch_loss.item())
                rec_losses.append(rec_loss.item())
                reg_losses.append(reg_loss.item())
                cl_losses.append(cl_loss.item())
                
                self.optimizer.zero_grad()
                batch_loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 4)
                self.optimizer.step()
            
            train_loss = np.mean(train_losses)
            rec_loss = np.mean(rec_losses)
            reg_loss = np.mean(reg_losses)
            cl_loss = np.mean(cl_losses)

            final_train_losses.append([ep, train_loss])
            final_rec_losses.append([ep,rec_loss])
            final_reg_losses.append([ep, reg_loss])
            final_cl_losses.append([ep, cl_loss])

            model.eval()
            with torch.no_grad():
                self.user_emb, self.item_emb = model()
            s_eval = time.time()
            self.fast_evaluation(ep)
            e_eval = time.time()
            print("Eval time: %f s" % (e_eval - s_eval))
        
            del dropped_adj1
            del dropped_adj2


        self.save_loss(final_train_losses, final_rec_losses, final_reg_losses, final_cl_losses)
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
        adj = data.interaction_mat
        # kg_adj = data.kg_interaction_mat
        self.adj  = TorchGraphInterface.convert_sparse_mat_to_tensor(adj).to_dense().to(device)
        # self.kg_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(kg_adj).to_dense().to(device)
        
        self._parse_args(config)
        self.embedding_dict = self._init_model()
        
        self.fc_u = nn.Linear(self.input_dim, self.hyper_dim)
        self.fc_i = nn.Linear(self.input_dim, self.hyper_dim)
        
        self.hgnn_u =  [HGNNConv(leaky=self.p, input_dim=self.hyper_dim, hyper_dim=self.hyper_dim) for _ in range(self.n_layers)]
        self.hgnn_i = [ HGNNConv(leaky=self.p, input_dim=self.hyper_dim, hyper_dim=self.hyper_dim) for _ in range(self.n_layers)]
        
        self.non_linear = nn.ReLU()
        self.dropout = nn.Dropout(self.drop_rate)

        self.apply(xavier_uniform_initialization)
        
    def _parse_args(self, config):

        self.lRate = float(config['learnRate'])
        self.lr_decay = float(config['learnRateDecay'])
        self.maxEpoch = int(config['num.max.epoch'])
        self.batchSize = int(config['batch_size'])
        self.reg = float(config['reg.lambda'])
        self.hyper_dim = int(config['hyper.size'])
        self.input_dim = int(config['input.size'])

        self.drop_rate = float(config['dropout'])
        self.p = float(config['leaky'])
        self.n_layers = int(config['gnn_layer'])
        self.aug_type = int(config['aug_type'])
        self.temp = float(config['temp'])

    def graph_reconstruction(self):
        if self.aug_type==0 or 1:
            dropped_adj = self.random_graph_augment()
        else:
            dropped_adj = []
            for k in range(self.n_layers):
                dropped_adj.append(self.random_graph_augment())
        return dropped_adj

    def random_graph_augment(self):
        dropped_mat = None
        if self.aug_type == 0:
            dropped_mat = GraphAugmentor.node_dropout(self.data.interaction_mat, self.drop_rate)
        elif self.aug_type == 1 or self.aug_type == 2:
            dropped_mat = GraphAugmentor.edge_dropout(self.data.interaction_mat, self.drop_rate)
        return TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat).to_dense().cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.input_dim)).to(device)),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.input_dim)).to(device))
        })
        return embedding_dict

    def cal_cl_loss(self, idxs, perturbed_mat1, perturbed_mat2):
        if type(idxs[0]) is not list:
            u_idx = torch.unique(idxs[0])
        else:
            u_idx = torch.unique(torch.Tensor(idxs[0]).cuda().type(torch.long))
        if type(idxs[1]) is not list:
            i_idx = torch.unique(idxs[1])
        else:
            i_idx = torch.unique(torch.Tensor(idxs[1]).cuda().type(torch.long))
        user_view_1, item_view_1 = self.forward(perturbed_mat1)
        user_view_2, item_view_2 = self.forward(perturbed_mat2)
        view1 = torch.cat((user_view_1[u_idx],item_view_1[i_idx]),0)
        view2 = torch.cat((user_view_2[u_idx],item_view_2[i_idx]),0)
        return InfoNCE(view1,view2,self.temp)

    def forward(self, perturbed_adj=None):
        uEmbed = self.embedding_dict['user_emb']
        iEmbed = self.embedding_dict['item_emb']
        
        uEmbed = self.dropout(self.non_linear(self.fc_u(uEmbed)))
        iEmbed = self.dropout(self.non_linear(self.fc_i(iEmbed)))
        
        embeds = torch.cat([uEmbed, iEmbed], 0)
        all_embeddings = [embeds]
        
        for k in range(self.n_layers):
            # ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            
            if perturbed_adj is not None:
                if isinstance(perturbed_adj, list):
                    hyperULat = self.hgnn_u[k](perturbed_adj[k], uEmbed)
                    hyperILat = self.hgnn_i[k](perturbed_adj[k].T, iEmbed)
                else:
                    hyperULat = self.hgnn_u[k](perturbed_adj, uEmbed)
                    hyperILat = self.hgnn_i[k](perturbed_adj.T, iEmbed)
            else:
                hyperULat = self.hgnn_u[k](self.adj, uEmbed)
                hyperILat = self.hgnn_i[k](self.adj.T, iEmbed)
            ego_embeddings = torch.cat([hyperULat, hyperILat], dim=0)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:self.data.user_num+ self.data.item_num]
        return user_all_embeddings, item_all_embeddings

class HGNNConv(nn.Module):
    def __init__(self, leaky, input_dim, hyper_dim, bias=False):
        super(HGNNConv, self).__init__()
        self.hyper_dim = hyper_dim
        self.act = nn.LeakyReLU(negative_slope=leaky).cuda()
        self.fc1 = nn.Linear(input_dim, hyper_dim ,bias=False).cuda() 
        self.fc2 = nn.Linear(hyper_dim, hyper_dim ,bias=False).cuda()  
        self.fc3 = nn.Linear(hyper_dim, hyper_dim ,bias=False).cuda()  
        
        self.ln1 = torch.nn.LayerNorm(hyper_dim).cuda()
        self.ln2 = torch.nn.LayerNorm(hyper_dim).cuda()
        self.ln3 = torch.nn.LayerNorm(hyper_dim).cuda()
        self.ln4 = torch.nn.LayerNorm(hyper_dim).cuda()
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(hyper_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # # self.apply(xavier_uniform_)
        # torch.nn.init.xavier_uniform_(self.weight)
        
    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight)
        if self.bias is not None:
            self.bias.data.data.fill_(0.)

    def forward(self, adj, embeds):
        lat1 = self.act(adj.T @ embeds)
        lat1 = self.ln1(lat1)

        lat2 = self.act(self.fc1(lat1)) +  lat1
        lat2 = self.ln2(lat2)
        
        lat3 = self.act(self.fc2(lat2)) + lat2
        lat3 = self.ln3(lat3)
        
        lat4 = self.act(self.fc3(lat3)) + lat3 
        output = adj @ lat4
        if self.bias is not None:
            output += self.bias 
        output = self.ln4(output)
        ret = self.act(output)
        return ret
    
