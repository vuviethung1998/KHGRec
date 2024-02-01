import time
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
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from util.loss_torch import bpr_loss, l2_reg_loss, contrastLoss
from util.init import *
import torch.nn.init as init 
from base.main_recommender import GraphRecommender
from util.evaluation import early_stopping
from util.sampler import next_batch_unified
from base.torch_interface import TorchGraphInterface

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class KHGRec(GraphRecommender):
    def __init__(self, conf, training_set, test_set, knowledge_set, **kwargs):
        GraphRecommender.__init__(self, conf, training_set, test_set, knowledge_set, **kwargs)

        self.device = torch.device(f"cuda:{kwargs['gpu_id']}" if torch.cuda.is_available() else 'cpu')
        self._parse_config( kwargs)
        self.set_seed()
        self.model = HGNNModel(self.data, self.data_kg, kwargs, self.device).to(self.device)
        
        self.attention_user = Attention(in_size=self.hyper_dim, hidden_size=self.hyper_dim).to(self.device)
        self.attention_item = Attention(in_size=self.hyper_dim, hidden_size=self.hyper_dim).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lRate, weight_decay=self.weight_decay)
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
        self.drop_rate = float(kwargs['drop_rate'])
        self.layers = int(kwargs['n_layers'])
        self.cl_rate = float(kwargs['cl_rate'])
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

    def train(self, load_pretrained=False):
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
            relations = list(self.data_kg.laplacian_dict.keys())

            train_model.train()
            s_train = time.time()
            
            for n, batch in enumerate(next_batch_unified(self.data, self.data_kg, self.batchSize, self.batchSizeKG, device=self.device)):
                user_idx, pos_idx, neg_idx, kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = batch
                # train KG
                ego_embed = train_model(mode='kg', keep_rate=1-self.drop_rate)
                user_emb_kg, item_emb_kg = ego_embed[train_model.user_indices], ego_embed[train_model.item_indices]
                train_model.update_attention(ego_embed, kg_batch_head, kg_batch_pos_tail, kg_batch_relation, relations)
                
                kg_batch_head_emb = ego_embed[kg_batch_head]
                kg_batch_pos_tail_emb = ego_embed[kg_batch_pos_tail]
                kg_batch_neg_tail_emb = ego_embed[kg_batch_neg_tail]

                # Train CF
                user_emb_cf, item_emb_cf = train_model(mode='cf', keep_rate=1-self.drop_rate)

                if self.use_attention:
                    item_emb_fused, _ = self.attention_item(torch.stack([item_emb_cf, item_emb_kg], dim=1))
                else:
                    item_emb_fused = torch.mean(torch.stack([item_emb_cf, item_emb_kg], dim=1), dim=1)
                    
                h_cf = torch.cat([user_emb_cf, item_emb_cf], dim=0)
                h_kg = torch.cat([user_emb_kg, item_emb_kg], dim=0)
                
                anchor_emb = user_emb_cf[user_idx]
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

            cf_loss = np.mean(cf_losses)
            kg_loss = np.mean(kg_losses)
            
            e_train = time.time() 
            train_time = e_train - s_train

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
                item_emb, _ = self.attention_item(torch.stack([item_emb_cf, item_emb_kg], dim=1))
                
                self.user_emb, self.item_emb = user_emb_cf, item_emb
                test_time, data_ep = self.fast_evaluation(ep, train_model)
            
                cur_recall =  float(data_ep[2].split(':')[1])
                recall_list.append(cur_recall)
                best_recall, should_stop = early_stopping(recall_list, self.early_stopping_steps)
                
                if should_stop:
                    break 

            self.save_performance_row(ep, train_time, test_time, data_ep)
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
        rec_loss = bpr_loss(anchor_emb, pos_emb, neg_emb)
        reg_loss = l2_reg_loss(self.reg, anchor_emb, pos_emb, neg_emb) / batch_size
        return rec_loss, reg_loss

class HGNNModel(nn.Module):
    def __init__(self, data, data_kg, args, device):
        super(HGNNModel, self).__init__()
        self.data = data
        self.data_kg = data_kg 
        
        self.device = device
        self.use_drop_edge = False
        self.sparse_norm_adj  = TorchGraphInterface.convert_sparse_mat_to_tensor(data.norm_adj).to(self.device)
        self.kg_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(data_kg.norm_kg_adj).to(self.device)
        kg_shape = self.kg_adj.shape
        self.att_adj = TorchGraphInterface.sparse_identity(kg_shape[0]).to(self.device)
        self.device = device 

        self.user_indices =  torch.LongTensor(list(data.user.keys())).to(self.device)
        self.item_indices = torch.LongTensor(list(data.item.keys())).to(self.device)
        
        self._parse_args(args)
        self.embedding_dict = self._init_model()
        self.hgnn_cf = []
        self.hgnn_kg  = []
        
        self.hgnn_layer_cf = SelfAwareEncoder(self.data, self.emb_size, self.hyper_size, self.layers, self.p, self.drop_rate, self.device)
        self.hgnn_layer_kg = RelationalAwareEncoder(self.p, self.drop_rate, self.layers, self.hyper_size)
        self.relation_emb = nn.Parameter(init.xavier_uniform_(torch.empty(self.data_kg.n_relations, self.input_dim))).to(self.device)
        self.trans_M = nn.Parameter(init.xavier_uniform_(torch.empty(self.data_kg.n_relations, self.hyper_dim, self.relation_dim))).to(self.device)
        self.act = nn.LeakyReLU(self.p)
        self.dropout = nn.Dropout(self.drop_rate)
        self.edgeDropper = SpAdjDropEdge()
        
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
        self.batchSize = int(args['batch_size'])
        self.batchSizeKG = int(args['batch_size_kg'])
        
    def _init_model(self):
        initializer = init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_entity_emb': nn.Parameter(initializer(torch.empty(self.data_kg.n_users_entities, self.hyper_dim)).to(self.device)),
        })
        return embedding_dict
    
    def calculate_cf_embeddings(self, keep_rate:float=1):
        uEmbed = self.embedding_dict['user_entity_emb'][self.user_indices] 
        iEmbed = self.embedding_dict['user_entity_emb'][self.item_indices]
        ego_embeddings = torch.cat([uEmbed, iEmbed], 0)
        sparse_norm_adj = self.edgeDropper(self.sparse_norm_adj, keep_rate)
        user_all_embeddings, item_all_embeddings = self.hgnn_layer_cf(ego_embeddings, sparse_norm_adj)
        return user_all_embeddings, item_all_embeddings
    
    def calculate_kg_embeddings(self, keep_rate: float=1):
        embeds = self.embedding_dict['user_entity_emb']
        sparse_norm_kg_adj = self.edgeDropper(self.kg_adj, keep_rate)
        hyperLat = self.hgnn_layer_kg(embeds, sparse_norm_kg_adj, att_adj=self.att_adj)
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
        self.att_adj.data = A_in.to(self.device)

    def forward(self, mode='cf', keep_rate=1):
        if mode == 'cf':
            user_embed, item_embed = self.calculate_cf_embeddings(keep_rate=keep_rate)
            return user_embed, item_embed
        elif mode == 'kg':
            entity_embed = self.calculate_kg_embeddings(keep_rate=keep_rate)
            return entity_embed 

    def calculate_cf_loss(self, anchor_emb, pos_emb, neg_emb, reg):
        rec_loss = bpr_loss(anchor_emb, pos_emb, neg_emb)
        reg_loss = l2_reg_loss(reg, anchor_emb, pos_emb, neg_emb) / self.batchSize
        cf_loss  = rec_loss + reg_loss
        return cf_loss
    
    def calculate_kg_loss(self, h_embed, r, pos_t_embed, neg_t_embed, reg_kg):
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

        reg_loss = l2_reg_loss(reg_kg, r_mul_h, r_embed, r_mul_pos_t, r_mul_neg_t) / self.batchSizeKG
        loss = kg_loss + reg_loss
        return loss
        
    def calculate_ssl_loss(self, data, ancs, poss, emb_cf, emb_kg, temp):
        embeds1 = emb_cf
        embeds2 = emb_kg
        sslLoss = contrastLoss(embeds1[:data.n_users], embeds2[:data.n_users], torch.unique(ancs), temp) + \
                    contrastLoss(embeds2[data.n_users:], embeds2[data.n_users:], torch.unique(poss), temp)
        return sslLoss

class SelfAwareEncoder(nn.Module):
    def __init__(self, data, emb_size, hyper_size, n_layers, leaky, drop_rate, device, use_self_att=False):
        super(SelfAwareEncoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.hyper_size = hyper_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.relu = nn.ReLU()
        self.act = nn.LeakyReLU(leaky)
        self.dropout = nn.Dropout(drop_rate)
        self.edgeDropper = SpAdjDropEdge()
        
        self.use_self_att = use_self_att

        self.hgnn_layers = torch.nn.ModuleList()
        self.ugformer_layers = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()

        for i in range(self.layers):
            encoder_layers = TransformerEncoderLayer(d_model=hyper_size, nhead=1, dim_feedforward=32, dropout=drop_rate) # Default batch_first=False (seq, batch, feature)
            enc_norm = nn.LayerNorm(hyper_size)
            self.ugformer_layers.append(TransformerEncoder(encoder_layers, 1, norm=enc_norm).to(device))
            self.hgnn_layers.append(HGCNConv(leaky=leaky))
            self.lns.append(torch.nn.LayerNorm(hyper_size))

    def forward(self, ego_embeddings, sparse_norm_adj):
        res = ego_embeddings
        all_embeddings = []
        for k in range(self.layers):
            if self.use_self_att:
                # self-attention over all nodes
                input_Tr = torch.unsqueeze(ego_embeddings, 1)  #[seq_length, batch_size=1, dim] for pytorch transformer
                input_Tr = self.ugformer_layers[k](input_Tr)
                ego_embeddings = torch.squeeze(input_Tr, 1)
            if k != self.layers - 1: 
                ego_embeddings = self.lns[k](self.hgnn_layers[k](sparse_norm_adj, ego_embeddings))  + res
            else:
                ego_embeddings = self.lns[k](self.hgnn_layers[k](sparse_norm_adj, ego_embeddings, act=False)) + res
            all_embeddings += [ego_embeddings]
        
        user_all_embeddings = all_embeddings[-1][:self.data.n_users]
        item_all_embeddings = all_embeddings[-1][self.data.n_users:]
        return user_all_embeddings, item_all_embeddings

class RelationalAwareEncoder(nn.Module):
    def __init__(self, leaky, dropout, n_layers, hyper_dim):
        super(RelationalAwareEncoder, self).__init__()
        self.leaky = leaky 
        self.dropout = dropout 
        self.n_layers = n_layers
        self.convs = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(AttHGCNConv(leaky=leaky))
            self.lns.append(torch.nn.LayerNorm(hyper_dim))
            
    def forward(self, embs, sparse_adj, att_adj):
        residual = embs
        for i, conv in enumerate(self.convs):
            if i != self.n_layers - 1:
                embs = self.lns[i](conv(sparse_adj, att_adj, embs)) + residual
            else:
                embs = self.lns[i](conv(sparse_adj, att_adj, embs, act=False)) + residual
        return embs 

class AttHGCNConv(nn.Module):
    def __init__(self, leaky):
        super(AttHGCNConv, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=leaky)

    def forward(self, inp_adj, att_adj, embs, act=True):
        adj = torch.sparse.mm(att_adj,inp_adj)
        if act:
            return self.act(torch.sparse.mm(adj,  torch.sparse.mm(adj.t(), embs)))
        else:
            return torch.sparse.mm(adj, torch.sparse.mm(adj.t(), embs))

class HGCNConv(nn.Module):
    def __init__(self, leaky):
        super(HGCNConv, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=leaky)

    def forward(self, adj, embs, act=True):
        if act:
            return self.act(torch.sparse.mm(adj, torch.sparse.mm(adj.t(), embs)))
        else:
            return torch.sparse.mm(adj, torch.sparse.mm(adj.t(), embs))

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
    
