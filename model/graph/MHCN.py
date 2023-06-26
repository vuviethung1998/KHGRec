from base.graph_recommender import GraphRecommender
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# from util.loss_tf import bpr_loss
from data.social import Relation
# from base.tf_interface import TFGraphInterface
from util.sampler import next_batch_pairwise
from util.conf import OptionConf
import torch
import torch.nn as nn 
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from util.loss_torch import bpr_loss, l2_reg_loss, EmbLoss
from util.init import *
from torch_geometric.nn import MessagePassing
from util.loss_torch import BPRLoss, EmbLoss
from base.torch_interface import TorchGraphInterface
import os
import numpy as np 
import time 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# paper: Self-Supervised Multi-Channel Hypergraph Convolutional Network for Social Recommendation. WWW'21
class MHCN(GraphRecommender):
    def __init__(self, conf, training_set, test_set, **kwargs):
        GraphRecommender.__init__(self, conf, training_set, test_set, **kwargs)
        args = OptionConf(self.config['MHCN'])
        # self.n_layers = int(args['-n_layer'])
        self.ss_rate = float(args['-ss_rate'])
        self.reg = float(self.config['reg.lambda'])
        
        self.kwargs = kwargs
        # self.social_data = Relation(conf, kwargs['social.data'], self.data.user)
        self.reg_loss = EmbLoss()
        self.model = MHCNModel(self.config, self.kwargs, self.data)
        
    def print_model_info(self):
        super(MHCN, self).print_model_info()
        # # print social relation statistics
        print('Social data size: (user number: %d, relation number: %d).' % (self.model.social_data.size()))
        print('=' * 80)
        
    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data,self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                model.train()
                # s_model = time.time()
                rec_user_emb, rec_item_emb = model()
                # e_model = time.time()
                # print("Model predicting time: %f s" % (e_model - s_model))
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                
                # s_loss  = time.time() 
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                ssl_loss = self.ss_rate * self.model.ssl_layer_loss()
                reg_loss = self.reg * self.reg_loss(user_emb, pos_item_emb, neg_item_emb)
                # e_loss = time.time()
                # print("Loss calculation time: %f s" % (e_loss - s_loss))
                
                # Backward and optimize
                batch_loss = rec_loss + reg_loss + ssl_loss 
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0 and n > 0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'ssl_loss', ssl_loss.item(), 'reg_loss', reg_loss.item())
            model.eval()
            with torch.no_grad():
                self.user_emb, self.item_emb = model()
                
            s_eval = time.time()
            self.fast_evaluation(epoch)
            e_eval = time.time()
            print("Eval time: %f s" % (e_eval - s_eval))
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        
    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model()
                    
    def predict(self, u):
        user_id  = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[user_id], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class MHCNModel(nn.Module):
    def __init__(self, config, args, data):
        super(MHCNModel, self).__init__()
        # social dataset 
        self.data = data 
        self.social_data = Relation(config, args['social.data'], self.data.user)
        
        # load dataset info
        R = self.data.normalize_graph_mat(self.data.interaction_mat)
        self.R =  TorchGraphInterface.convert_sparse_mat_to_tensor(R).cuda()
        H_s, H_j, H_p = self.build_hyper_adj_mats()
        self.H_s = TorchGraphInterface.convert_sparse_mat_to_tensor(H_s).cuda()
        self.H_j = TorchGraphInterface.convert_sparse_mat_to_tensor(H_j).cuda()
        self.H_p = TorchGraphInterface.convert_sparse_mat_to_tensor(H_p).cuda()
        
        # load parameters info
        args = OptionConf(config['MHCN'])
        self.n_layers = int(args['-n_layer'])
        self.embedding_size = int(config['embedding.size'])

        # define embedding and loss
        self.user_embedding = nn.Embedding(self.data.user_num, self.embedding_size)
        self.item_embedding = nn.Embedding(self.data.item_num, self.embedding_size)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
            
        # define gating layers
        self.gating_c1 = GatingLayer(self.embedding_size)
        self.gating_c2 = GatingLayer(self.embedding_size)
        self.gating_c3 = GatingLayer(self.embedding_size)
        self.gating_simple = GatingLayer(self.embedding_size)
        
        # define self supervised gating layers
        self.ss_gating_c1 = GatingLayer(self.embedding_size)
        self.ss_gating_c2 = GatingLayer(self.embedding_size)
        self.ss_gating_c3 = GatingLayer(self.embedding_size)

        # define attention layers
        self.attention_layer = AttLayer(self.embedding_size)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        
    def build_hyper_adj_mats(self):
        S = self.social_data.get_social_mat()
        Y = self.data.interaction_mat 
        B = S.multiply(S.T)
        U = S - B
        C1 = (U.dot(U)).multiply(U.T)
        A1 = C1 + C1.T
        C2 = (B.dot(U)).multiply(U.T) + (U.dot(B)).multiply(U.T) + (U.dot(U)).multiply(B)
        A2 = C2 + C2.T
        C3 = (B.dot(B)).multiply(U) + (B.dot(U)).multiply(B) + (U.dot(B)).multiply(B)
        A3 = C3 + C3.T
        A4 = (B.dot(B)).multiply(B)
        C5 = (U.dot(U)).multiply(U) + (U.dot(U.T)).multiply(U) + (U.T.dot(U)).multiply(U)
        A5 = C5 + C5.T
        A6 = (U.dot(B)).multiply(U) + (B.dot(U.T)).multiply(U.T) + (U.T.dot(U)).multiply(B)
        A7 = (U.T.dot(B)).multiply(U.T) + (B.dot(U)).multiply(U) + (U.dot(U.T)).multiply(B)
        A8 = (Y.dot(Y.T)).multiply(B)
        A9 = (Y.dot(Y.T)).multiply(U)
        A9 = A9 + A9.T
        A10  = Y.dot(Y.T) - A8 - A9
        # addition and row-normalization
        H_s = sum([A1, A2, A3, A4, A5, A6, A7])
        # add epsilon to avoid divide by zero Warning
        H_s = H_s.multiply(1.0 / (H_s.sum(axis=1) + 1e-7).reshape(-1, 1))
        H_j = sum([A8, A9])
        H_j = H_j.multiply(1.0 / (H_j.sum(axis=1) + 1e-7).reshape(-1, 1))
        H_p = A10
        H_p = H_p.multiply(H_p > 1)
        H_p = H_p.multiply(1.0 / (H_p.sum(axis=1) + 1e-7).reshape(-1, 1))
        return H_s, H_j, H_p

    def forward(self):
        # get ego embeddings
        user_embeddings = self.user_embedding.weight.cuda()
        item_embeddings = self.item_embedding.weight.cuda()

        # self-gating
        user_embeddings_c1 = self.gating_c1(user_embeddings)
        user_embeddings_c2 = self.gating_c2(user_embeddings)
        user_embeddings_c3 = self.gating_c3(user_embeddings)
        simple_user_embeddings = self.gating_simple(user_embeddings)

        all_embeddings_c1 = [user_embeddings_c1]
        all_embeddings_c2 = [user_embeddings_c2]
        all_embeddings_c3 = [user_embeddings_c3]
        all_embeddings_simple = [simple_user_embeddings]
        all_embeddings_i = [item_embeddings]

        for layer_idx in range(self.n_layers):
            mixed_embedding = self.attention_layer(user_embeddings_c1, user_embeddings_c2, user_embeddings_c3) + simple_user_embeddings / 2
            # Channel S
            user_embeddings_c1 = torch.spmm(self.H_s, user_embeddings_c1)
            norm_embeddings = F.normalize(user_embeddings_c1, p=2, dim=1)
            all_embeddings_c1 += [norm_embeddings]
            # Channel J
            user_embeddings_c2 = torch.spmm(self.H_j, user_embeddings_c2)
            norm_embeddings = F.normalize(user_embeddings_c2, p=2, dim=1)
            all_embeddings_c2 += [norm_embeddings]
            # Channel P
            user_embeddings_c3 = torch.spmm(self.H_p, user_embeddings_c3)
            norm_embeddings = F.normalize(user_embeddings_c3, p=2, dim=1)
            all_embeddings_c3 += [norm_embeddings]
            
            # item convolution
            new_item_embeddings = torch.spmm(torch.transpose(self.R, 0, 1), mixed_embedding)
            norm_embeddings = F.normalize(new_item_embeddings, p=2, dim=1)
            all_embeddings_i += [norm_embeddings]
            simple_user_embeddings = torch.spmm(self.R, item_embeddings)
            all_embeddings_simple += [F.normalize(simple_user_embeddings, p=2, dim=1)]
            item_embeddings = new_item_embeddings
            
        # averaging the channel-specific embeddings
        user_embeddings_c1 = torch.stack(all_embeddings_c1, dim=0).sum(dim=0)
        user_embeddings_c2 = torch.stack(all_embeddings_c2, dim=0).sum(dim=0)
        user_embeddings_c3 = torch.stack(all_embeddings_c3, dim=0).sum(dim=0)
        simple_user_embeddings = torch.stack(all_embeddings_simple, dim=0).sum(dim=0)
        item_all_embeddings = torch.stack(all_embeddings_i, dim=0).sum(dim=0)

        # aggregating channel-specific embeddings
        user_all_embeddings = self.attention_layer(user_embeddings_c1, user_embeddings_c2, user_embeddings_c3)
        user_all_embeddings += simple_user_embeddings / 2

        # assign
        self.final_item_embeddings= item_all_embeddings
        self.final_user_embeddings = user_all_embeddings
        
        return user_all_embeddings, item_all_embeddings

    def hierarchical_self_supervision(self, user_embeddings, adj_mat):
        def row_shuffle(embedding):
            shuffled_embeddings = embedding[torch.randperm(embedding.size(0))]
            return shuffled_embeddings
        def row_column_shuffle(embedding):
            shuffled_embeddings = embedding[:, torch.randperm(embedding.size(1))]
            shuffled_embeddings = shuffled_embeddings[torch.randperm(embedding.size(0))]
            return shuffled_embeddings
        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), dim=1)

        # For Douban, normalization is needed.
        user_embeddings = F.normalize(user_embeddings, p=2, dim=1) 
        edge_embeddings = torch.spmm(adj_mat, user_embeddings)
        # Local MIM
        pos = score(user_embeddings, edge_embeddings)
        neg1 = score(row_shuffle(user_embeddings), edge_embeddings)
        neg2 = score(row_column_shuffle(edge_embeddings), user_embeddings)
        local_loss = torch.sum(-torch.log(torch.sigmoid(pos - neg1)) - torch.log(torch.sigmoid(neg1 - neg2)))
        # Global MIM
        graph = torch.mean(edge_embeddings, dim=0, keepdim=True)
        pos = score(edge_embeddings, graph)
        neg1 = score(row_column_shuffle(edge_embeddings), graph)
        global_loss = torch.sum(-torch.log(torch.sigmoid(pos - neg1)))
        return global_loss + local_loss

    def ssl_layer_loss(self):
        ss_loss = self.hierarchical_self_supervision(self.ss_gating_c1(self.final_user_embeddings), self.H_s)
        ss_loss += self.hierarchical_self_supervision(self.ss_gating_c2(self.final_user_embeddings), self.H_j)
        ss_loss += self.hierarchical_self_supervision(self.ss_gating_c3(self.final_user_embeddings), self.H_p)
        return ss_loss         

class GatingLayer(nn.Module):
    def __init__(self, dim):
        super(GatingLayer, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(self.dim, self.dim)
        self.activation = nn.Sigmoid()

    def forward(self, emb):
        embedding = self.linear(emb)
        embedding = self.activation(embedding)
        embedding = torch.mul(emb, embedding)
        return embedding
    
class BipartiteGCNConv(MessagePassing):
    def __init__(self, dim):
        super(BipartiteGCNConv, self).__init__(aggr='add')
        self.dim = dim

    def forward(self, x, edge_index, edge_weight, size):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.dim)

class AttLayer(nn.Module):
    def __init__(self, dim):
        super(AttLayer, self).__init__()
        self.dim = dim
        self.attention_mat = nn.Parameter(torch.randn([self.dim, self.dim]))
        self.attention = nn.Parameter(torch.randn([1, self.dim]))

    def forward(self, *embs):
        weights = []
        emb_list = []
        for embedding in embs:
            weights.append(torch.sum(torch.mul(self.attention, torch.matmul(embedding, self.attention_mat)), dim=1))
            emb_list.append(embedding)
        score = torch.nn.Softmax(dim=0)(torch.stack(weights, dim=0))
        embeddings = torch.stack(emb_list, dim=0)
        scores = score.unsqueeze(dim=2).repeat(1, 1, self.dim)
        mixed_embeddings = torch.mul(embeddings, scores).sum(dim=0)
        return mixed_embeddings
