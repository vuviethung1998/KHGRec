import math
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.loss_torch import L2_loss_mean
import time
import os, sys
from os.path import abspath
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random 
import numpy as np 
from random import shuffle,randint,choice,sample

from base.kggraph_recommender import KGGraphRecommender
from util.knowledge_sampler import next_batch_pairwise, next_batch_kg
from util.loss_torch import bpr_loss, l2_reg_loss, EmbLoss, contrastLoss
from util.init import *
from base.torch_interface import TorchGraphInterface
from util.conf import OptionConf
from util.algorithm import find_k_largest

class KGAT(KGGraphRecommender):
    def __init__(self, conf, training_set, test_set, knowledge_set, **kwargs):
        super(KGAT, self).__init__(conf, training_set, test_set, knowledge_set, **kwargs)
        self._parse_args(kwargs)
        A_in = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data_kg.kg_interaction_mat).cuda()
        self.model = KGATEncoder(kwargs, self.data_kg, A_in=A_in)

    def _parse_args(self, args):
        self.lRate = args['lrate']
        self.lRateKG = args['lratekg']
        self.maxEpoch = args['max_epoch']
        self.batchSize = args['batch_size']
        self.batchSizeKG = args['batch_size_kg']
        self.dataset = args['dataset']
        self.embed_dim = args['embedding_size']
        self.relation_dim = args['relation_dim']
        self.n_layers = args['n_layers']
        self.lr_decay = args['lr_decay']

        self.reg = args['reg']
        self.reg_kg = args['reg_kg']
        self.aggregation_type = args['aggregation_type']
        self.mess_dropout = args['mess_dropout']
        self.conv_dim_list = args['conv_dim_list']
        self.seed = args['seed']
        self.alpha = args['alpha']

    def train(self):
        # seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
        lst_train_losses = []
        lst_rec_losses = []
        lst_kg_losses = []
        lst_performances = []
        
        cf_optimizer  = torch.optim.Adam(self.model.parameters(), lr=self.lRate)
        kg_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lRateKG)

        kg_data = self.data_kg.kg_train_data.to_numpy()
        kg_dict = self.data_kg.train_kg_dict
            
        for ep in range(self.maxEpoch):
            self.model.train()
            
            train_losses = []
            cf_losses = []
            kg_losses = []
            
            cf_total_loss = 0
            kg_total_loss = 0
            
            n_cf_batch = int(self.data.n_cf_train // self.batchSize + 1)
            n_kg_batch = int(self.data_kg.n_kg_train // self.batchSizeKG + 1)
            
            shuffle(kg_data)
            
            # Learn cf graph
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batchSize)):
                user_idx, pos_idx, neg_idx = batch
                entity_emb = self.model.calc_cf_embeddings()
                
                user_emb = entity_emb[user_idx]
                pos_item_emb = entity_emb[pos_idx]
                neg_item_emb = entity_emb[neg_idx]
                
                cf_batch_loss = self.model.calc_cf_loss(user_emb, pos_item_emb, neg_item_emb)
                if np.isnan(cf_batch_loss.cpu().detach().numpy()):
                    print('ERROR (CF Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(ep, n, n_cf_batch))

                cf_batch_loss.backward()
                cf_optimizer.step()
                cf_optimizer.zero_grad()
                cf_total_loss += cf_batch_loss.item()
                
                cf_losses.append(cf_batch_loss.item())
                if (n % 20) == 0:
                    print('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(ep, n, n_cf_batch,  cf_batch_loss.item(), cf_total_loss / (n+1)))
            
            # Learn knowledge grap
            for n, batch in enumerate(next_batch_kg(kg_data, kg_dict, self.batchSizeKG)):
                kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = batch
                
                kg_batch_loss = self.model.calc_kg_loss(kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail)
                if np.isnan(kg_batch_loss.cpu().detach().numpy()):
                    print('ERROR (KG Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(ep, n, n_kg_batch))
                kg_batch_loss.backward()
                kg_optimizer.step()
                kg_optimizer.zero_grad()
                kg_total_loss += kg_batch_loss.item()
                
                kg_losses.append(kg_batch_loss.item())
                if (n % 10) == 0:
                    print('KG Training: Epoch {:04d} Iter {:04d} / {:04d} | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(ep, n, n_kg_batch,  kg_batch_loss.item(), kg_total_loss / (n+1)))

            # Learn attention 
            h_list = self.data_kg.h_list.cuda()
            t_list = self.data_kg.t_list.cuda()
            r_list = self.data_kg.r_list.cuda()
            relations = list(self.data_kg.laplacian_dict.keys())
            self.model.update_attention(h_list, t_list, r_list, relations)
            self.model.eval()

            cf_loss = np.mean(cf_losses)
            kg_loss = np.mean(kg_losses)
            train_loss = cf_loss + kg_loss

            with torch.no_grad():
                entity_emb = self.model.calc_cf_embeddings()
                user_emb = entity_emb[self.model.user_indices]
                item_emb = entity_emb[self.model.item_indices]
                data_ep = self.fast_evaluation(self.model, ep, user_emb, item_emb)
            self.save_performance_row(ep, data_ep)
            self.save_loss_row([ep, train_loss, cf_loss, kg_loss])
        
            lst_performances.append(data_ep)
            lst_train_losses.append([ep, train_loss]) 
            lst_rec_losses.append([ep, cf_loss])
            lst_kg_losses.append([ep, kg_loss])
        self.save_loss(lst_train_losses, lst_rec_losses, lst_kg_losses)
        self.save_perfomance_training(lst_performances)
        user_emb, item_emb = self.best_user_emb, self.best_item_emb
        return user_emb, item_emb  

    def test(self, user_emb, item_emb):
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
            sys.stdout.write(r)
            sys.stdout.flush()

        # predict
        rec_list = {}
        user_count = len(self.data.test_set)
        for i, user in enumerate(self.data.test_set):
            user_id = self.data_kg.u2id[user]
            score = torch.matmul(user_emb[user_id], item_emb.transpose(0, 1))
            candidates = score.cpu().numpy()
            
            rated_list, li = self.data.user_rated(user)
            for item in rated_list:
                candidates[self.data_kg.i2id[item]] = -10e8
            # s_find_k_largest = time.time()
            ids, scores = find_k_largest(self.max_N, candidates)

            item_names = [self.data_kg.id2i[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
            if i % 1000 == 0:
                process_bar(i, user_count)
        process_bar(user_count, user_count)
        print('')
        self.evaluate(rec_list) 

    def predict(self, u):
        user_id = self.data_kg.u2id[u]
        score = torch.matmul(self.user_emb[user_id], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class Aggregator(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, aggregator_type):
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()

        if self.aggregator_type == 'gcn':
            self.linear = nn.Linear(self.in_dim, self.out_dim)       # W in Equation (6)
            # nn.init.xavier_uniform_(self.linear.weight)

        elif self.aggregator_type == 'graphsage':
            self.linear = nn.Linear(self.in_dim * 2, self.out_dim)   # W in Equation (7)
            # nn.init.xavier_uniform_(self.linear.weight)

        elif self.aggregator_type == 'bi-interaction':
            
            self.linear1 = nn.Linear(self.in_dim, self.out_dim)      # W1 in Equation (8)
            self.linear2 = nn.Linear(self.in_dim, self.out_dim)      # W2 in Equation (8)
            # nn.init.xavier_uniform_(self.linear1.weight)
            # nn.init.xavier_uniform_(self.linear2.weight)
        else:
            raise NotImplementedError
        self.ln1 = nn.LayerNorm(self.out_dim)
        self.ln2 = nn.LayerNorm(self.out_dim)
        
    def forward(self, ego_embeddings, A_in):
        """
        ego_embeddings:  (n_users + n_entities, in_dim)
        A_in:            (n_users + n_entities, n_users + n_entities), torch.sparse.FloatTensor
        """
        # Equation (3)
        side_embeddings = torch.matmul(A_in, ego_embeddings)

        if self.aggregator_type == 'gcn':
            # Equation (6) & (9)
            embeddings = ego_embeddings + side_embeddings
            embeddings = self.ln1(self.activation(self.linear(embeddings)))
            
        elif self.aggregator_type == 'graphsage':
            # Equation (7) & (9)
            embeddings = torch.cat([ego_embeddings, side_embeddings], dim=1)
            embeddings = self.ln1(self.activation(self.linear(embeddings)))

        elif self.aggregator_type == 'bi-interaction':
            # Equation (8) & (9)
            sum_embeddings = self.ln1(self.activation(self.linear1(ego_embeddings + side_embeddings)))
            bi_embeddings = self.ln2(self.activation(self.linear2(ego_embeddings * side_embeddings)))
            embeddings = bi_embeddings + sum_embeddings

        embeddings = self.message_dropout(embeddings)           # (n_users + n_entities, out_dim)
        return embeddings

class KGATEncoder(nn.Module):

    def __init__(self, args, data_kg, A_in=None, user_pre_embed=None, item_pre_embed=None):

        super(KGATEncoder, self).__init__()
        
        self.user_indices = torch.LongTensor(list(data_kg.userent.keys())).cuda()
        self.item_indices =  torch.LongTensor(list(data_kg.itement.keys())).cuda()
        
        self.use_pretrain = 1
        
        self.n_users = data_kg.n_users
        self.n_entities = data_kg.n_entities
        self.n_relations = data_kg.n_relations
        self.n_users_entities = data_kg.n_users_entities
        self.embed_dim = args['embedding_size']
        self.relation_dim = args['relation_dim']

        self.aggregation_type = args['aggregation_type']
        self.conv_dim_list = [args['embedding_size']] + eval(args['conv_dim_list'])
        self.mess_dropout = eval(args['mess_dropout'])
        self.n_layers = len(eval(args['conv_dim_list']))

        self.kg_l2loss_lambda = args['reg_kg']
        self.cf_l2loss_lambda = args['reg']

        self.entity_user_embed = nn.Embedding(self.n_users_entities, self.embed_dim).cuda()
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim).cuda()
        self.trans_M = nn.Parameter(torch.Tensor(self.n_relations, self.embed_dim, self.relation_dim)).cuda()

        self.all_user_idx = list(data_kg.userent.keys())
        self.all_item_idx =  list(data_kg.itement.keys())
        
        if (self.use_pretrain == 1) and (user_pre_embed is not None) and (item_pre_embed is not None):
            other_entity_embed = nn.Parameter(torch.Tensor(self.n_entities - item_pre_embed.shape[0], self.embed_dim))
            nn.init.xavier_uniform_(other_entity_embed)
            entity_user_embed = torch.cat([item_pre_embed, other_entity_embed, user_pre_embed], dim=0)
            self.entity_user_embed.weight = nn.Parameter(entity_user_embed)
        else:
            nn.init.xavier_uniform_(self.entity_user_embed.weight)

        nn.init.xavier_uniform_(self.relation_embed.weight)
        nn.init.xavier_uniform_(self.trans_M)

        self.aggregator_layers = nn.ModuleList()
        for k in range(self.n_layers):
            self.aggregator_layers.append(Aggregator(self.conv_dim_list[k], self.conv_dim_list[k + 1], self.mess_dropout[k], self.aggregation_type).cuda())

        self.A_in = nn.Parameter(torch.sparse.FloatTensor(self.n_users_entities, self.n_users_entities))
        if A_in is not None:
            self.A_in.data = A_in 
        self.A_in.requires_grad = False

    def calc_cf_embeddings(self):
        ego_embed = self.entity_user_embed.weight
        all_embed = [ego_embed]

        for idx, layer in enumerate(self.aggregator_layers):
            ego_embed = layer(ego_embed, self.A_in.cuda())
            norm_embed = F.normalize(ego_embed, p=2, dim=1)
            all_embed.append(norm_embed)

        # Equation (11)
        all_embed = torch.cat(all_embed, dim=1)         # (n_users + n_entities, concat_dim)
        return all_embed

    def calc_cf_loss(self, user_embed, item_pos_embed, item_neg_embed):
        """
        user_ids:       (cf_batch_size)
        item_pos_ids:   (cf_batch_size)
        item_neg_ids:   (cf_batch_size)
        """
        # Equation (12)
        pos_score = torch.sum(user_embed * item_pos_embed, dim=1)   # (cf_batch_size)
        neg_score = torch.sum(user_embed * item_neg_embed, dim=1)   # (cf_batch_size)

        # Equation (13)
        # cf_loss = F.softplus(neg_score - pos_score)
        cf_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        cf_loss = torch.mean(cf_loss)

        l2_loss = L2_loss_mean(user_embed) + L2_loss_mean(item_pos_embed) + L2_loss_mean(item_neg_embed)
        loss = cf_loss + self.cf_l2loss_lambda * l2_loss
        return loss

    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relation_embed(r)                                                # (kg_batch_size, relation_dim)
        W_r = self.trans_M[r]                                                           # (kg_batch_size, embed_dim, relation_dim)

        h_embed = self.entity_user_embed(h)                                             # (kg_batch_size, embed_dim)
        pos_t_embed = self.entity_user_embed(pos_t)                                     # (kg_batch_size, embed_dim)
        neg_t_embed = self.entity_user_embed(neg_t)                                     # (kg_batch_size, embed_dim)

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

        l2_loss = L2_loss_mean(r_mul_h) + L2_loss_mean(r_embed) + L2_loss_mean(r_mul_pos_t) + L2_loss_mean(r_mul_neg_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss

    def update_attention_batch(self, h_list, t_list, r_idx):
        r_embed = self.relation_embed.weight[r_idx]
        W_r = self.trans_M[r_idx]

        h_embed = self.entity_user_embed.weight[h_list]
        t_embed = self.entity_user_embed.weight[t_list]

        # Equation (4)
        r_mul_h = torch.matmul(h_embed, W_r)
        r_mul_t = torch.matmul(t_embed, W_r)
        v_list = torch.sum(r_mul_t * torch.tanh(r_mul_h + r_embed), dim=1)
        return v_list

    def update_attention(self, h_list, t_list, r_list, relations):
        device = self.A_in.device

        rows = []
        cols = []
        values = []

        for r_idx in relations:
            index_list = torch.where(r_list == r_idx)
            batch_h_list = h_list[index_list]
            batch_t_list = t_list[index_list]

            batch_v_list = self.update_attention_batch(batch_h_list, batch_t_list, r_idx)
            rows.append(batch_h_list)
            cols.append(batch_t_list)
            values.append(batch_v_list)

        rows = torch.cat(rows)
        cols = torch.cat(cols)
        values = torch.cat(values)

        indices = torch.stack([rows, cols])
        shape = self.A_in.shape
        A_in = torch.sparse.FloatTensor(indices, values, torch.Size(shape))

        # Equation (5)
        A_in = torch.sparse.softmax(A_in.cpu(), dim=1)
        self.A_in.data = A_in.to(device)

    def calc_score(self, user_ids, item_ids):
        """
        user_ids:  (n_users)
        item_ids:  (n_items)
        """
        all_embed = self.calc_cf_embeddings()           # (n_users + n_entities, concat_dim)
        user_embed = all_embed[user_ids]                # (n_users, concat_dim)
        item_embed = all_embed[item_ids]                # (n_items, concat_dim)

        # Equation (12)
        cf_score = torch.matmul(user_embed, item_embed.transpose(0, 1))    # (n_users, n_items)
        return cf_score 
    