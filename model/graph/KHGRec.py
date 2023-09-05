import torch
import torch.nn as nn
import numpy as np 
import pandas as pd 
import random 
import os 
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from base.main_recommender import GraphRecommender
from util.sampler import next_batch_unified
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, contrastLoss
from util.evaluation import early_stopping

class KHGRec(GraphRecommender):
    def __init__(self, conf, training_set, test_set, knowledge_set, **kwargs):
        super(KHGRec, self).__init__(conf, training_set, test_set, knowledge_set, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self._parse_config( kwargs)
        self.set_seed()

        self.model = KHGRec_Encoder(self.data, self.data_kg, kwargs).to(self.device)
        self.attention_user = Attention(in_size=self.hyper_dim, hidden_size=self.hyper_dim).to(self.device)
        self.attention_item = Attention(in_size=self.hyper_dim, hidden_size=self.hyper_dim).to(self.device)

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
        self.drop_rate = float(kwargs['drop_rate'])
        self.cl_rate = float(kwargs['cl_rate'])
        self.temp = kwargs['temp']
        self.seed = kwargs['seed']
        self.early_stopping_steps = kwargs['early_stopping_steps']
        self.weight_decay = kwargs['weight_decay']
        self.mode = kwargs['mode']
        
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
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=self.lr_decay, patience=5)
        
        recall_list = []
        lst_train_losses = []
        lst_cf_losses = []
        lst_kg_losses = []
        lst_cl_losses = []
        lst_performances = []

        for epoch in range(self.maxEpoch):
            cf_losses = []
            kg_losses = []
            cl_losses = []
            
            n_cf_batch = int(self.data.n_cf_train // self.batchSize + 1)

            cf_total_loss = 0
            kg_total_loss = 0
            cl_total_loss = 0 
            
            for n, batch in enumerate(next_batch_unified(self.data, self.data_kg, self.batchSize, self.batchSizeKG, device=self.device)):
                user_idx, pos_idx, neg_idx, kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = batch
                user_emb_cf, item_emb_cf = model(keep_rate=1-self.drop_rate, mode='cf')
                
                ego_embed = model(keep_rate=1-self.drop_rate, mode='kg')
                user_emb_kg, item_emb_kg = ego_embed[model.user_indices], ego_embed[model.item_indices]
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
                
                cf_batch_loss = model.calculate_cf_loss(anchor_emb, pos_emb, neg_emb)
                kg_batch_loss = model.calculate_kg_loss(kg_batch_head_emb, kg_batch_relation, kg_batch_pos_tail_emb, kg_batch_neg_tail_emb)
                cf_total_loss += cf_batch_loss.item()
                kg_total_loss +=  kg_batch_loss.item()  

                if self.use_contrastive:
                    cl_batch_loss = model.calculate_ssl_loss(self.data, user_idx, pos_idx, h_cf, h_kg)
                    cl_losses.append(cl_batch_loss.item())
                    cl_total_loss += cl_batch_loss.item()
                    batch_loss = cf_batch_loss + kg_batch_loss + cl_batch_loss
                else:
                    cl_batch_loss = 0
                    batch_loss = cf_batch_loss + kg_batch_loss 

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                cf_losses.append(cf_batch_loss.item())
                kg_losses.append(kg_batch_loss.item())
                if (n % 20) == 0:
                    if self.use_contrastive:
                        print('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, n, n_cf_batch,  cf_batch_loss.item(), cf_total_loss / (n+1)))
                        print('KG Training: Epoch {:04d} Iter {:04d} / {:04d} | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, n, n_cf_batch, kg_batch_loss.item(), kg_total_loss / (n+1)))
                        print('CL Training: Epoch {:04d} Iter {:04d} / {:04d} | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, n, n_cf_batch, cl_batch_loss.item(), cl_total_loss / (n+1)))
                    else:                                        
                        print('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, n, n_cf_batch,  cf_batch_loss.item(), cf_total_loss / (n+1)))
                        print('KG Training: Epoch {:04d} Iter {:04d} / {:04d} | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, n, n_cf_batch, kg_batch_loss.item(), kg_total_loss / (n+1)))
                        
                h_list  = self.data_kg.h_list.to(self.device)
                t_list  = self.data_kg.t_list.to(self.device)
                r_list = self.data_kg.r_list.to(self.device)
                relations = list(self.data_kg.laplacian_dict.keys())
                model.update_attention(ego_embed, h_list, t_list, r_list, relations)
            
            cf_loss = np.mean(cf_losses)
            kg_loss = np.mean(kg_losses)

            if self.use_contrastive:
                cl_loss = np.mean(cl_losses)
                train_loss = cf_loss + kg_loss + cl_loss 
            else:
                cl_loss  = 0
                train_loss = cf_loss + kg_loss

            lst_cf_losses.append([epoch,cf_loss])
            lst_kg_losses.append([epoch, kg_loss])
            lst_train_losses.append([epoch, train_loss])
            lst_cl_losses.append([epoch, cl_loss])
            scheduler.step(train_loss)

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
        self.save_loss(lst_train_losses, lst_cf_losses, lst_kg_losses, lst_cl_losses)
        self.save_perfomance_training(lst_performances)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model()
            print("Saving")
            self.save_model(self.model)     

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class KHGRec_Encoder(nn.Module):
    def __init__(self, data, data_kg, kwargs):
        super(KHGRec_Encoder, self).__init__()
        self.data = data
        self.data_kg = data_kg
        self._parse_config(kwargs)
        self.embedding_dict = self._init_model()

        self.user_indices =  torch.LongTensor(list(data.user.keys())).cuda()
        self.item_indices = torch.LongTensor(list(data.item.keys())).cuda()

        self.relation_emb =  nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.data_kg.n_relations, self.input_dim))).cuda()
        self.trans_M = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.data_kg.n_relations, self.hyper_dim, self.relation_dim))).cuda()

        self.norm_adj = data.norm_adj
        self.norm_kg_adj = data_kg.norm_kg_adj
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
        self.sparse_norm_kg_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_kg_adj).cuda()
        self.relu = nn.ReLU()
        self.act = nn.LeakyReLU(self.leaky)
        self.dropout = nn.Dropout(self.drop_rate)
        self.edgeDropper = SpAdjDropEdge()

    def _parse_config(self, kwargs):
        self.dataset = kwargs['dataset']
        self.lRate = float(kwargs['lrate'])
        self.lr_decay = float(kwargs['lr_decay'])
        self.maxEpoch = int(kwargs['max_epoch'])
        self.batchSize = int(kwargs['batch_size'])
        self.batchSizeKG = int(kwargs['batch_size_kg'])
        self.reg = float(kwargs['reg'])
        self.reg_kg = float(kwargs['reg_kg'])

        self.input_dim =int(kwargs['embedding_size'])
        self.hyper_dim = int(kwargs['hyper_dim'])
        self.relation_dim = int(kwargs['relation_dim'])

        self.leaky = float(kwargs['p'])
        self.drop_rate = float(kwargs['drop_rate'])
        self.layers = int(kwargs['n_layers'])
        self.cl_rate = float(kwargs['cl_rate'])
        self.temp = kwargs['temp']
        self.early_stopping_steps = kwargs['early_stopping_steps']

        self.mode = kwargs['mode']
        
        if self.mode == 'full':
            self.use_contrastive = True
            self.use_attention = True
        elif self.mode == 'wo_attention':
            self.use_contrastive = True
            self.use_attention = False
        elif self.mode == 'wo_ssl':
            self.use_contrastive = False
            self.use_attention = True 

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.n_users, self.input_dim))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.n_items, self.input_dim))),
            'entity_emb': nn.Parameter(initializer(torch.empty(self.data_kg.n_users_entities, self.input_dim))),
            'user_w': nn.Parameter(initializer(torch.empty(self.input_dim, self.hyper_dim))),
            'item_w': nn.Parameter(initializer(torch.empty(self.input_dim, self.hyper_dim))),
            'entity_w': nn.Parameter(initializer(torch.empty(self.input_dim, self.hyper_dim))),
        })
        return embedding_dict
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, keep_rate=1, mode='cf'):
        if mode == 'cf':
            self.user_emb =  self.embedding_dict['user_emb'] @ self.embedding_dict['user_w']
            self.item_emb = self.embedding_dict['item_emb'] @ self.embedding_dict['item_w']  
            ego_embeddings = torch.cat([self.user_emb, self.item_emb], 0)

            all_embeddings = [ego_embeddings]
            for k in range(self.layers):
                self.sparse_norm_adj = self.edgeDropper(self.sparse_norm_adj, keep_rate)
                prev_embeddings = ego_embeddings 
                if k != self.layers - 1: 
                    ego_embeddings = self.act(torch.sparse.mm(self.sparse_norm_adj, torch.sparse.mm(self.sparse_norm_adj.t(), ego_embeddings)))
                else:
                    ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, torch.sparse.mm(self.sparse_norm_adj.t(), ego_embeddings))
                ego_embeddings += prev_embeddings
                all_embeddings += [ego_embeddings]

            user_all_embeddings = all_embeddings[-1][:self.data.n_users]
            item_all_embeddings = all_embeddings[-1][self.data.n_users:]
            return user_all_embeddings, item_all_embeddings
        elif mode == 'kg':
            self.entity_emb = self.embedding_dict['entity_emb'] @ self.embedding_dict['entity_w']  
            ego_embeddings = self.entity_emb 
            all_embeddings = [ego_embeddings] 
            for k in range(self.layers):
                self.sparse_norm_kg_adj = self.edgeDropper(self.sparse_norm_kg_adj, keep_rate)
                prev_embeddings = ego_embeddings 
                if k != self.layers - 1: 
                    ego_embeddings = self.act(torch.sparse.mm(self.sparse_norm_kg_adj, torch.sparse.mm(self.sparse_norm_kg_adj.t(), ego_embeddings)))
                else:
                    ego_embeddings = torch.sparse.mm(self.sparse_norm_kg_adj, torch.sparse.mm(self.sparse_norm_kg_adj.t(), ego_embeddings))
                ego_embeddings += prev_embeddings
                all_embeddings += [ego_embeddings]
            return all_embeddings[-1]
    
    def calculate_cf_loss(self, anchor_emb, pos_emb, neg_emb):
        rec_loss = bpr_loss(anchor_emb, pos_emb, neg_emb)
        reg_loss =  l2_reg_loss(self.reg, anchor_emb, pos_emb, neg_emb) / self.batchSize
        cf_loss  = rec_loss + reg_loss
        return cf_loss

    def calculate_kg_loss(self, h_embed, r, pos_t_embed, neg_t_embed):
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

        reg_loss = l2_reg_loss(self.reg, r_mul_h, r_embed, r_mul_pos_t, r_mul_neg_t) / self.batchSizeKG
        loss = kg_loss + reg_loss
        return loss
    
    def calculate_ssl_loss(self, data, ancs, poss, emb_cf, emb_kg):
        embeds1 = emb_cf
        embeds2 = emb_kg
        sslLoss = contrastLoss(embeds1[:data.n_users], embeds2[:data.n_users], torch.unique(ancs), self.temp) + \
                    contrastLoss(embeds2[data.n_users:], embeds2[data.n_users:], torch.unique(poss), self.temp)
        sslLoss *= self.cl_rate 
        return sslLoss
    
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
        shape = self.sparse_norm_kg_adj.shape
        A_in = torch.sparse.FloatTensor(indices, values, torch.Size(shape))
        # Equation (5)
        A_in = torch.sparse.softmax(A_in.cpu(), dim=1)
        self.sparse_norm_kg_adj.data = A_in.cuda()

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