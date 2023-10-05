from tqdm import tqdm
import os 
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.nn.init as init
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random 
from torch_geometric.nn import HypergraphConv

from util.loss_torch import bpr_loss, EmbLoss, contrastLoss
from util.init import *
from base.torch_interface import TorchGraphInterface
import torch.nn.init as init 
from data.augmentor import GraphAugmentor
from base.main_recommender import GraphRecommender
from util.evaluation import early_stopping
from util.sampler import next_batch_unified
from data.augmentor import drop_edges

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class HGNN(GraphRecommender):
    def __init__(self, conf, training_set, test_set, knowledge_set, **kwargs):
        GraphRecommender.__init__(self, conf, training_set, test_set, knowledge_set, **kwargs)

        self.reg_loss = EmbLoss() 
        
        self.device = torch.device(f"cuda:{kwargs['gpu_id']}" if torch.cuda.is_available() else 'cpu')
        self._parse_config( kwargs)
        self.set_seed()
        self.model = HGNNModel(self.data, self.data_kg, kwargs, self.device).to(self.device)
        
        self.attention_user = Attention(in_size=self.hyper_dim, hidden_size=self.hyper_dim).to(self.device)
        self.attention_item = Attention(in_size=self.hyper_dim, hidden_size=self.hyper_dim).to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lRate, weight_decay=self.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=self.lr_decay, patience=10)

    def _parse_config(self, kwargs):
        self.dataset = kwargs['dataset']
        
        self.lRate = float(kwargs['lrate'])
        self.lr_decay = float(kwargs['lr_decay'])
        self.maxEpoch = int(kwargs['max_epoch'])
        self.batchSize = int(kwargs['batch_size'])
        self.batchSizeKG = int(kwargs['batch_size_kg'])
        self.reg = float(kwargs['reg'])
        self.reg_kg = float(kwargs['reg_kg'])
        self.hyper_dim = int(kwargs['hyper_dim'])
        self.p = float(kwargs['p'])
        self.layers = int(kwargs['n_layers'])
        self.cl_rate = float(kwargs['cl_rate'])
        # self.use_contrastive = kwargs['use_contrastive']
        # self.use_attention = kwargs['use_attention']
        self.temp = kwargs['temp']
        self.seed = kwargs['seed']
        self.mode = kwargs['mode']
        self.early_stopping_steps = kwargs['early_stopping_steps']
        self.weight_decay = kwargs['weight_decay']

        if self.mode == 'full':
            self.use_contrastive = True
            self.use_attention = True
        elif self.mode == 'wo_attention':
            self.use_contrastive = True
            self.use_attention = False
        elif self.mode == 'wo_ssl':
            self.use_contrastive = False
            self.use_attention = True 
            
    def set_seed(self):
        seed = self.seed 
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"Random seed set as {seed}")

    def train(self):
        print("start training")
        train_model = self.model 
        lst_train_losses = []
        lst_cf_losses = []
        lst_kg_losses = []
        lst_cl_losses = [] 
        
        lst_performances = []
        recall_list = []
        
        for ep in range(self.maxEpoch):        
            cf_losses = []
            kg_losses = []
            cl_losses = [] 
            
            cf_total_loss = 0
            kg_total_loss = 0
            cl_total_loss = 0 

            n_cf_batch = int(self.data.n_cf_train // self.batchSize + 1)
            n_kg_batch = int(self.data_kg.n_kg_train // self.batchSizeKG + 1)

            train_model.train()
            for n, batch in enumerate(next_batch_unified(self.data, self.data_kg, self.batchSize, self.batchSizeKG, device=self.device)):
                user_idx, pos_idx, neg_idx, kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = batch
                user_emb_cf, item_emb_cf = train_model(mode='cf', keep_rate=0.5)

                ego_embed = train_model(mode='kg')
                user_emb_kg, item_emb_kg = ego_embed[train_model.user_indices], ego_embed[train_model.item_indices]
                
                kg_batch_head_emb = ego_embed[kg_batch_head]
                kg_batch_pos_tail_emb = ego_embed[kg_batch_pos_tail]
                kg_batch_neg_tail_emb = ego_embed[kg_batch_neg_tail]
                
                if self.use_attention:
                    user_emb_fused, _ = self.attention_user(torch.stack([user_emb_cf, user_emb_kg], dim=1))
                    item_emb_fused, _ = self.attention_item(torch.stack([item_emb_cf, item_emb_kg], dim=1))
                else:
                    user_emb_fused = torch.mean(torch.stack([user_emb_cf, user_emb_kg], dim=1), dim=1)
                    item_emb_fused = torch.mean(torch.stack([item_emb_cf, item_emb_kg], dim=1), dim=1)
                h_cf = torch.cat([user_emb_cf, item_emb_cf], dim=0)
                h_kg = torch.cat([user_emb_kg, item_emb_kg], dim=0)
                
                anchor_emb = user_emb_fused[user_idx]
                pos_emb = item_emb_fused[pos_idx]
                neg_emb = item_emb_fused[neg_idx]

                cf_batch_loss = train_model.calculate_cf_loss(anchor_emb, pos_emb, neg_emb, self.reg)
                kg_batch_loss = train_model.calculate_kg_loss(kg_batch_head_emb, kg_batch_relation, kg_batch_pos_tail_emb, kg_batch_neg_tail_emb, self.reg_kg)
                cf_total_loss += cf_batch_loss.item()
                kg_total_loss +=  kg_batch_loss.item()
                
                if self.use_contrastive:
                    cl_batch_loss = self.cl_rate * train_model.calculate_ssl_loss(self.data, user_idx, pos_idx, h_cf, h_kg, self.temp)
                    cl_losses.append(cl_batch_loss.item())
                    cl_total_loss += cl_batch_loss.item()
                    batch_loss = cf_batch_loss + kg_batch_loss + cl_batch_loss
                else:
                    cl_batch_loss = 0
                    batch_loss = cf_batch_loss + kg_batch_loss 

                self.optimizer.zero_grad()
                torch.nn.utils.clip_grad_norm_(train_model.parameters(), 4)
                batch_loss.backward()
                self.optimizer.step()

                cf_losses.append(cf_batch_loss.item())
                kg_losses.append(kg_batch_loss.item())
                if (n % 20) == 0:
                    if self.use_contrastive:
                        print('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(ep, n, n_cf_batch,  cf_batch_loss.item(), cf_total_loss / (n+1)))
                        print('KG Training: Epoch {:04d} Iter {:04d} / {:04d} | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(ep, n, n_cf_batch, kg_batch_loss.item(), kg_total_loss / (n+1)))
                        print('CL Training: Epoch {:04d} Iter {:04d} / {:04d} | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(ep, n, n_cf_batch, cl_batch_loss.item(), cl_total_loss / (n+1)))
                    else:                                        
                        print('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(ep, n, n_cf_batch,  cf_batch_loss.item(), cf_total_loss / (n+1)))
                        print('KG Training: Epoch {:04d} Iter {:04d} / {:04d} | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(ep, n, n_cf_batch, kg_batch_loss.item(), kg_total_loss / (n+1)))
                        
                # h_list  = self.data_kg.h_list.to(self.device)
                # t_list  = self.data_kg.t_list.to(self.device)
                # r_list = self.data_kg.r_list.to(self.device)
                # relations = list(self.data_kg.laplacian_dict.keys())
                # train_model.update_attention(ego_embed, h_list, t_list, r_list, relations)
            
            cf_loss = np.mean(cf_losses)
            kg_loss = np.mean(kg_losses)

            if self.use_contrastive:
                cl_loss = np.mean(cl_losses)
                train_loss = cf_loss + kg_loss + cl_loss 
            else:
                cl_loss  = 0
                train_loss = cf_loss + kg_loss

            lst_cf_losses.append([ep,cf_loss])
            lst_kg_losses.append([ep, kg_loss])
            lst_train_losses.append([ep, train_loss])
            lst_cl_losses.append([ep, cl_loss])
            
            self.scheduler.step(train_loss)
            
            # Evaluation
            train_model.eval()
            self.attention_user.eval()
            self.attention_item.eval()
            
            with torch.no_grad():
                user_emb_cf, item_emb_cf = train_model(mode='cf')
                ego_emb = train_model(mode='kg')
                user_emb_kg, item_emb_kg = ego_emb[train_model.user_indices], ego_emb[train_model.item_indices]

                user_emb, _ = self.attention_user(torch.stack([user_emb_cf, user_emb_kg], dim=1))
                item_emb, _ = self.attention_item(torch.stack([item_emb_cf, item_emb_kg], dim=1))
                
                self.user_emb, self.item_emb = user_emb, item_emb
                data_ep = self.fast_evaluation(ep, train_model)
            
                cur_recall =  float(data_ep[2].split(':')[1])
                recall_list.append(cur_recall)
                best_recall, should_stop = early_stopping(recall_list, self.early_stopping_steps)
                
                if should_stop:
                    break 

            self.save_performance_row(ep, data_ep)
            self.save_loss_row([ep, train_loss, cf_loss, kg_loss, cl_loss])
            lst_performances.append(data_ep)

        self.save_loss(lst_train_losses, lst_cf_losses, lst_kg_losses, lst_cl_losses)
        self.save_perfomance_training(lst_performances)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
    
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
    def __init__(self, data, data_kg, args, device):
        super(HGNNModel, self).__init__()
        self.data = data
        self.data_kg = data_kg 
        
        self.device = device
        self.use_drop_edge = False
        # self.sparse_norm_adj  = TorchGraphInterface.convert_sparse_mat_to_tensor(data.norm_adj).to(self.device)
        # self.kg_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(data_kg.kg_interaction_mat).to(self.device)
        
        self.device = device 

        self.edge_index = data.edge_index.cuda()
        self.edge_index_t = data.edge_index_t.cuda()
        self.edge_index_kg = data_kg.edge_index_kg.cuda()

        self.user_indices =  torch.LongTensor(list(data.user.keys())).to(self.device)
        self.item_indices = torch.LongTensor(list(data.item.keys())).to(self.device)
        
        self._parse_args(args)
        self.embedding_dict = self._init_model()

        self.hgnn_cf = []
        self.hgnn_kg  = []
        
        self.hgnn_layer_u = SelfAwareHGCNConv(leaky=self.p, dropout=self.drop_rate, n_layers=self.layers, nheads=self.nheads, input_dim=self.emb_size, hidden_dim=64, hyper_dim=self.hyper_size, bias=True).cuda()
        self.hgnn_layer_i = SelfAwareHGCNConv(leaky=self.p, dropout=self.drop_rate, n_layers=self.layers, nheads=self.nheads, input_dim=self.emb_size, hidden_dim=64, hyper_dim=self.hyper_size, bias=True).cuda()
        self.hgnn_layer_kg = RelationalAwareHGCNConv(leaky=self.p, dropout=self.drop_rate, n_layers=self.layers, nheads=self.nheads, input_dim=self.emb_size, hidden_dim=64, hyper_dim=self.hyper_size, bias=True).cuda()
        
        self.relation_emb = nn.Parameter(init.xavier_uniform_(torch.empty(self.data_kg.n_relations, self.input_dim))).to(self.device)
        self.trans_M = nn.Parameter(init.xavier_uniform_(torch.empty(self.data_kg.n_relations, self.hyper_dim, self.relation_dim))).to(self.device)

        self.act = nn.LeakyReLU(self.p)
        self.dropout = nn.Dropout(self.drop_rate)
        
    def _parse_args(self, args):
        self.input_dim = args['input_dim']
        self.hyper_dim = args['hyper_dim']
        self.p = args['p']
        self.drop_rate = args['drop_rate'] 
        self.layers = args['n_layers']
        self.temp = args['temp']
        self.aug_type = args['aug_type']
        self.relation_dim = args['relation_dim']
        self.nheads = int(args['nheads'])
        self.emb_size =  int(args['input_dim'])
        self.hyper_size =  int(args['hyper_dim'])
        
    def _init_model(self):
        initializer = init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_entity_emb': nn.Parameter(initializer(torch.empty(self.data_kg.n_users_entities, self.input_dim)).to(self.device)),
            'entity_emb': nn.Parameter(initializer(torch.empty(self.data_kg.n_relations, self.input_dim)).to(self.device))
        })
        return embedding_dict
    
    def calculate_cf_embeddings(self, keep_rate:float=1):
        uEmbed = self.embedding_dict['user_entity_emb'][self.user_indices] 
        iEmbed = self.embedding_dict['user_entity_emb'][self.item_indices]
        if self.use_drop_edge:
            self.edge_index = drop_edges(self.edge_index, aug_ratio=1-keep_rate)
            self.edge_index_t = drop_edges(self.edge_index_t, aug_ratio=1-keep_rate)
        ego_embeddings = torch.cat([uEmbed, iEmbed], 0)

        hyperLat1 = self.hgnn_layer_u(ego_embeddings, self.edge_index)
        hyperLat2 = self.hgnn_layer_i(ego_embeddings, self.edge_index_t)

        hyperULat = hyperLat1[:self.data.n_users]
        hyperILat = hyperLat2[self.data.n_users:]
        return hyperULat, hyperILat

    def calculate_kg_embeddings(self):
        embeds = self.embedding_dict['user_entity_emb']
        hyperLat = self.hgnn_layer_kg(embeds, self.edge_index_kg)
        return hyperLat 

    def update_attention_batch(self, ego_embed, h_list, t_list, r_idx):
        r_embed = self.relation_emb[r_idx]
        W_r = self.trans_M[r_idx]
        h_embed = ego_embed[h_list]
        t_embed = ego_embed[t_list]
        # Equation (4)
        r_mul_h = torch.matmul(h_embed, W_r)
        r_mul_t = torch.matmul(t_embed, W_r)
        v_list = torch.sum(r_mul_t * torch.tanh(r_mul_h + r_embed), dim=1)
        return v_list
    
    def update_attention(self, ego_embed, h_list, t_list, r_list, relations):
        rows, cols, values = [], [], []

        for r_idx in relations:
            index_list = torch.where(r_list == r_idx)
            batch_h_list = h_list[index_list]
            batch_t_list = t_list[index_list]
            batch_v_list = self.update_attention_batch(ego_embed, batch_h_list, batch_t_list, r_idx)
            rows.append(batch_h_list)
            cols.append(batch_t_list)
            values.append(batch_v_list)
        
        rows = torch.cat(rows)
        cols = torch.cat(cols)
        values = torch.cat(values)

        indices = torch.stack([rows, cols])
        shape = self.kg_adj.shape
        A_in = torch.sparse.FloatTensor(indices, values, torch.Size(shape))
        # Equation (5)
        A_in = torch.sparse.softmax(A_in.cpu(), dim=1)
        self.kg_adj.data = A_in.to(self.device)

    def forward(self, mode='cf', keep_rate=1):
        if mode == 'cf':
            user_embed, item_embed = self.calculate_cf_embeddings(keep_rate=keep_rate)
            return user_embed, item_embed
        elif mode == 'kg':
            entity_embed = self.calculate_kg_embeddings()
            return entity_embed 

    def calculate_cf_loss(self, anchor_emb, pos_emb, neg_emb, reg):
        calc_reg_loss = EmbLoss()
        rec_loss = bpr_loss(anchor_emb, pos_emb, neg_emb)
        reg_loss = reg * calc_reg_loss(anchor_emb, pos_emb, neg_emb)

        cf_loss  = rec_loss + reg_loss
        return cf_loss
    
    def calculate_kg_loss(self, h_embed, r, pos_t_embed, neg_t_embed, reg_kg):
        calc_reg_loss = EmbLoss()
        
        r_embed = self.relation_emb[r]                                                # (kg_batch_size, relation_dim)
        W_r = self.trans_M[r]                                                           # (kg_batch_size, embed_dim, relation_dim)
        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)                       # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)               # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)               # (kg_batch_size, relation_dim)

        # Equation (1)
        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)     # (kg_batch_size)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)     # (kg_batch_size)

        # Equation (2)
        # kg_loss = F.softplus(pos_score - neg_score)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        reg_loss =  reg_kg * calc_reg_loss(r_mul_h, r_embed, r_mul_pos_t, r_mul_neg_t)
        loss = kg_loss + reg_loss
        return loss
        
    def calculate_ssl_loss(self, data, ancs, poss, emb_cf, emb_kg, temp):
        embeds1 = emb_cf
        embeds2 = emb_kg
        sslLoss = contrastLoss(embeds1[:data.n_users], embeds2[:data.n_users], torch.unique(ancs), temp) + \
                    contrastLoss(embeds2[data.n_users:], embeds2[data.n_users:], torch.unique(poss), temp)
        return sslLoss

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

        for i in range(n_layers):
            first_channels = input_dim if i == 0 else hidden_dim
            second_channels = hyper_dim if i == n_layers - 1 else hidden_dim
            self.convs.append(HypergraphConv(first_channels, second_channels, use_attention=False, heads=nheads, attention_mode=att_mode,\
                                            concat=False, negative_slope=leaky, dropout=dropout, bias=bias))
            self.lns.append(torch.nn.LayerNorm(second_channels))
            self.residuals.append(nn.Linear(input_dim, second_channels).cuda())
            self.hyperedge_fc.append(nn.Linear(input_dim, first_channels).cuda())

    def forward(self, inp, adj, hyperedge_attr=None):
        embs = inp
        for i, conv in enumerate(self.convs):
            residual = self.residuals[i](inp)
            # if i == 0:
            #     hyperedge_attr_ = hyperedge_attr
            # else:
            #     hyperedge_attr_ = self.relu(self.hyperedge_fc[i](hyperedge_attr))
            # if i != self.n_layers - 1:
            #     embs = self.act(conv(embs, adj, hyperedge_attr=hyperedge_attr_)) + residual
            # else:
            #     embs = conv(embs, adj, hyperedge_attr=hyperedge_attr_) + residual
            if i != self.n_layers - 1:
                embs = self.act(self.lns[i](conv(embs, adj))) + residual
            else:
                embs = self.lns[i](conv(embs, adj)) + residual
        return embs 


class RelationalAwareHGCNConv(nn.Module):
    def __init__(self, leaky, dropout, n_layers, nheads, input_dim, hidden_dim, hyper_dim, att_mode='node', bias=True):
        super(RelationalAwareHGCNConv, self).__init__()

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

<<<<<<< HEAD:model/graph/HGNN_v3.py
=======

>>>>>>> 6dd1200558986a3f62071f90335ac8b50271caf0:test_model/HGNN_v2.py
        for i in range(n_layers):
            first_channels = input_dim if i == 0 else hidden_dim
            second_channels = hyper_dim if i == n_layers - 1 else hidden_dim
            self.convs.append(HypergraphConv(first_channels, second_channels, use_attention=False, heads=nheads, attention_mode=att_mode,\
                                            concat=False, negative_slope=leaky, dropout=dropout, bias=bias))
            self.lns.append(torch.nn.LayerNorm(second_channels))
            self.residuals.append(nn.Linear(input_dim, second_channels).cuda())
            self.hyperedge_fc.append(nn.Linear(input_dim, first_channels).cuda())

<<<<<<< HEAD:model/graph/HGNN_v3.py
    def relation_aware_attention(self, head_ent_embs, tail_ent_embs, rel_embs, adj):
        # item_embs: N, dim
        # entity_embs: N, e_num, dim
        # relations: N, e_num, r_dim
        # adj: N, e_num
        
        # N, e_num, dim
        Wh = head_ent_embs.unsqueeze(1).expand(tail_ent_embs.size())
        # N, e_num, dim
        We = tail_ent_embs
        a_input = torch.cat((Wh,We),dim=-1) # (N, e_num, 2*dim)
        # N,e,2dim -> N,e,dim
        e_input = torch.multiply(self.fc(a_input), rel_embs).sum(-1) # N,e
        e = self.act(e_input) # (N, e_num)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training) # N, e_num
        # (N, 1, e_num) * (N, e_num, out_features) -> N, out_features
        entity_emb_weighted = torch.bmm(attention.unsqueeze(1), tail_ent_embs).squeeze()
        entity_emb = entity_emb_weighted + head_ent_embs
        return entity_emb

    def forward(self, entity_embs, adj):
        embs = entity_embs

=======
    def forward(self, inp, adj, hyperedge_attr=None):
        embs = inp
>>>>>>> 6dd1200558986a3f62071f90335ac8b50271caf0:test_model/HGNN_v2.py
        for i, conv in enumerate(self.convs):
            residual = self.residuals[i](entity_embs)
            if i != self.n_layers - 1:
                embs = self.act(self.lns[i](conv(embs, adj))) + residual
            else:
                embs = self.lns[i](conv(embs, adj)) + residual
        return embs

class Attention(nn.Module):
    # This class module is a simple attention layer.
    def __init__(self, in_size, hidden_size=128):
        super(Attention, self).__init__()

        self.project = torch.nn.Sequential(
            torch.nn.Linear(in_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, hidden_size, bias=False),
        )

    def forward(self, z):
        w = self.project(z)  # (N, 2, D)
        beta = torch.softmax(w, dim=1)  # (N, 2, D)
        return (beta * z).sum(1), beta  # (N, D), (N, 2, D)

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
    