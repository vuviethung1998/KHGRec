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

from util.loss_torch import bpr_loss, EmbLoss, contrastLoss
from util.init import *
from base.torch_interface import TorchGraphInterface
import torch.nn.init as init 
from data.augmentor import GraphAugmentor
from base.main_recommender import GraphRecommender
from util.evaluation import early_stopping
from util.sampler import next_batch_unified

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
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lRate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=self.lr_decay, patience=5)

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
                user_emb_cf, item_emb_cf = train_model(mode='cf')

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
                        
                h_list  = self.data_kg.h_list.to(self.device)
                t_list  = self.data_kg.t_list.to(self.device)
                r_list = self.data_kg.r_list.to(self.device)
                relations = list(self.data_kg.laplacian_dict.keys())
                train_model.update_attention(ego_embed, h_list, t_list, r_list, relations)
            
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
        adj = data.interaction_mat
        self.data_kg = data_kg 
        kg_adj = data_kg.kg_interaction_mat
        
        self.device = device
        self.adj  = TorchGraphInterface.convert_sparse_mat_to_tensor(adj).to(self.device)
        self.kg_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(kg_adj).to(self.device)
        
        self.device = device 
        self.user_indices =  torch.LongTensor(list(data.user.keys())).to(self.device)
        self.item_indices = torch.LongTensor(list(data.item.keys())).to(self.device)
        
        self._parse_args(args)
        self.embedding_dict = self._init_model()
        
        self.fc_u = nn.Linear(self.input_dim, self.hyper_dim)
        self.fc_i = nn.Linear(self.input_dim, self.hyper_dim)
        self.fc_e = nn.Linear(self.input_dim, self.hyper_dim)
        
        self.hgnn_u = [HGNNConv(leaky=self.p, input_dim=self.hyper_dim, hyper_dim=self.hyper_dim, device=self.device) for i in range(self.layers)]
        self.hgnn_i = [HGNNConv(leaky=self.p, input_dim=self.hyper_dim, hyper_dim=self.hyper_dim, device=self.device) for i in range(self.layers) ] 
        self.hgnn_e = [HGNNConv(leaky=self.p, input_dim=self.hyper_dim, hyper_dim=self.hyper_dim, device=self.device) for i in range(self.layers) ] 
        
        self.relation_emb =   nn.Parameter(init.xavier_uniform_(torch.empty(self.data_kg.n_relations, self.input_dim))).to(self.device)
        self.trans_M = nn.Parameter(init.xavier_uniform_(torch.empty(self.data_kg.n_relations, self.hyper_dim, self.relation_dim))).to(self.device)

        self.non_linear = nn.ReLU()
        self.act = nn.LeakyReLU(self.p)
        self.dropout = nn.Dropout(self.drop_rate)
        self.apply(self._init_weights)
        
    def _parse_args(self, args):
        self.input_dim = args['input_dim']
        self.hyper_dim = args['hyper_dim']
        self.p = args['p']
        self.drop_rate = args['drop_rate'] 
        self.layers = args['n_layers']
        self.temp = args['temp']
        self.aug_type = args['aug_type']
        self.relation_dim = args['relation_dim']
        
    def _init_model(self):
        initializer = init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_entity_emb': nn.Parameter(initializer(torch.empty(self.data_kg.n_users_entities, self.input_dim)).to(self.device)),
        })
        return embedding_dict
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                init.zeros_(m.bias)

    def graph_reconstruction(self):
        if self.aug_type==0 or 1:
            dropped_adj = self.random_graph_augment()
        else:
            dropped_adj = [], []
            for k in range(self.n_layers):
                dropped_adj_ = self.random_graph_augment()
                dropped_adj.append(dropped_adj_)
        return dropped_adj


    def random_graph_augment(self):
        dropped_mat = None
        if self.aug_type == 0:
            dropped_mat = GraphAugmentor.node_dropout(self.data.interaction_mat, self.drop_rate)
        elif self.aug_type == 1 or self.aug_type == 2:
            dropped_mat = GraphAugmentor.edge_dropout(self.data.interaction_mat, self.drop_rate)
        return TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat).to(self.device)

    def calculate_cf_embeddings(self, perturbed_adj=None):
        uEmbed = self.embedding_dict['user_entity_emb'][self.user_indices]
        iEmbed = self.embedding_dict['user_entity_emb'][self.item_indices]
        
        uEmbed = self.dropout(self.act(self.fc_u(uEmbed)))
        iEmbed = self.dropout(self.act(self.fc_i(iEmbed)))
        
        embeds = torch.cat([uEmbed, iEmbed], 0)
        all_embeddings = [embeds]
        
        for k in range(self.layers):     
            if perturbed_adj is not None:
                if isinstance(perturbed_adj, list):
                    hyperULat = self.hgnn_u[k](perturbed_adj[k], uEmbed)
                    hyperILat = self.hgnn_i[k](perturbed_adj[k].t(), iEmbed)
                else:
                    hyperULat = self.hgnn_u[k](perturbed_adj, uEmbed)
                    hyperILat = self.hgnn_i[k](perturbed_adj.t(), iEmbed)
            else:
                hyperULat = self.hgnn_u[k](self.adj, uEmbed)
                hyperILat = self.hgnn_i[k](self.adj.t(), iEmbed)
            
            uEmbed += hyperULat
            iEmbed += hyperILat
            ego_embeddings = torch.cat([hyperULat, hyperILat], dim=0)
            all_embeddings += [ego_embeddings]
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.data.n_users]
        item_all_embeddings = all_embeddings[self.data.n_users:]
        return user_all_embeddings, item_all_embeddings 

    def calculate_kg_embeddings(self, perturbed_adj=None):
        eEmbed = self.embedding_dict['user_entity_emb']
        eEmbed = self.dropout(self.act(self.fc_e(eEmbed)))
        all_embeddings = [eEmbed]
        for k in range(self.layers):     
            if perturbed_adj is not None:
                if isinstance(perturbed_adj, list):
                    hyperELat = self.hgnn_e[k](perturbed_adj[k], eEmbed)
                else:
                    hyperELat = self.hgnn_e[k](perturbed_adj, eEmbed)
            else:
                hyperELat = self.hgnn_e[k](self.kg_adj, eEmbed)
            eEmbed += hyperELat
            all_embeddings += [eEmbed]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        return all_embeddings 

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

    def forward(self, perturbed_adj=None, mode='cf'):
        if mode == 'cf':
            user_embed, item_embed = self.calculate_cf_embeddings(perturbed_adj)
            return user_embed, item_embed
        elif mode == 'kg':
            entity_embed = self.calculate_kg_embeddings(perturbed_adj)
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

class HGNNConv(nn.Module):
    def __init__(self, leaky, input_dim, hyper_dim, device):
        super(HGNNConv, self).__init__()
        self.hyper_dim = hyper_dim
        self.act = nn.LeakyReLU(negative_slope=leaky).to(device)
        self.fc = nn.Linear(input_dim, hyper_dim ,bias=False).to(device) 
        
        self.ln1 = torch.nn.LayerNorm(hyper_dim).to(device)
        self.ln2 = torch.nn.LayerNorm(hyper_dim).to(device)
        
    def forward(self, adj, embeds):
        lat1 = self.ln1(self.fc(torch.spmm(adj.t(), embeds)))
        output = self.act((self.ln2(torch.spmm(adj, lat1))))
        return output
    
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