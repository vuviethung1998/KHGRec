import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import argparse
import numpy as np

from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise_kg, next_batch_pairwise
from util.conf import OptionConf
import torch
import torch.nn as nn 
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from util.loss_torch import bpr_loss, l2_reg_loss, EmbLoss, contrastLoss
from util.init import *
from base.torch_interface import TorchGraphInterface
import os
import numpy as np 
import time 
from torch.optim.lr_scheduler import ReduceLROnPlateau

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RippleNet(GraphRecommender):
    def __init__(self, conf, training_set, test_set, **kwargs):
        GraphRecommender.__init__(self, conf, training_set, test_set, **kwargs)

        self.reg = float(self.config['reg.lambda'])
        
        self.kwargs = kwargs
        self.reg_loss = EmbLoss()
        self.model = RippleNetModel(self.config, self.data)
    
    def train(self):
        model = self.model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-3, lr=self.lRate)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=self.lr_decay,patience=7)
        
        for epoch in range(self.maxEpoch):
            # --------------------hccf + ripple----------------------------
            train_losses =  []

            for n, batch in enumerate(next_batch_pairwise_kg(self.config, self.data,self.batch_size)):
                user_idx, pos_idx, neg_idx,memories_h, memories_r, memories_t  = batch
                kwargs = {}
                kwargs['users']  = user_idx
                kwargs['pos_items'] = pos_idx 
                kwargs['neg_items'] = neg_idx 
                kwargs['memories_h'] = memories_h 
                kwargs['memories_r'] = memories_r 
                kwargs['memories_t'] = memories_t 

                model.train()
                user_emb, item_emb, h_emb_list, t_emb_list, r_emb_list = model(kwargs, keep_rate=float(self.config['keep_rate']))

                ancEmbds = user_emb[user_idx]
                posEmbds = item_emb[pos_idx]
                negEmbds = item_emb[neg_idx]

                rec_loss,  reg_loss, kge_loss, ssl_loss = self.calculate_loss(ancEmbds, posEmbds, negEmbds, h_emb_list, t_emb_list, r_emb_list,user_idx, pos_idx)
                batch_loss = rec_loss + reg_loss + ssl_loss + kge_loss 

                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 4)
                optimizer.step()

                train_losses.append(batch_loss.item())

            batch_train_loss = np.mean(train_losses)
            scheduler.step(batch_train_loss)

            model.eval()
            with torch.no_grad():
                self.user_emb, self.item_emb, _, _, _ = model(kwargs,keep_rate=1)
            
            s_eval = time.time()

            self.fast_evaluation(epoch, kwargs)
            #--------------------------------------------
            e_eval = time.time()
            print("Eval time: %f s" % (e_eval - s_eval))
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
    
    def calculate_loss_hccf(self, anc_emb, pos_emb, neg_emb, user_idx, pos_idx):
        bprLoss = bpr_loss(anc_emb, pos_emb, neg_emb)
        reg_loss = self.reg * self.reg_loss(anc_emb, pos_emb, neg_emb)

        sslLoss = self.model.calculate_ssl_loss(user_idx, pos_idx)
        ssl_loss = sslLoss * self.ss_rate 
        return bprLoss, reg_loss, ssl_loss 

    def save(self, kwargs=None):
        with torch.no_grad():
            #--------------------hccf + ripple----------------------------
            self.best_user_emb, self.best_item_emb, _, _, _ = self.model(kwargs, keep_rate=1)
            #------------------------------------------------------------

    def predict(self, u):
        user_id  = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[user_id], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()  


class RippleNetModel(nn.Module):
    def __init__(self, args, data):
        super(RippleNetModel, self).__init__()
        initializer = nn.init.xavier_normal_

        self.data = data
        self._parse_args(args)

        self.uEmbed = nn.Parameter(initializer(torch.empty(self.data.user_num, self.dim).to(device)))
        self.iEmbed = nn.Parameter(initializer(torch.empty(self.data.item_num, self.dim).to(device)))

        self.adj_mat = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.interaction_mat).to(device)
        self.tp_adj_mat = torch.transpose(self.adj_mat, 0,1)

        self.hyperULat_layers = nn.ModuleList()
        self.hyperILat_layers = nn.ModuleList()
        self.weight_layers = nn.ModuleList()

        self.uLat_layers = nn.ModuleList()
        self.iLat_layers = nn.ModuleList()

        self.entity_emb = nn.Embedding(self.data.entity_num, self.dim)
        self.relation_emb = nn.Embedding(self.data.relation_num, self.dim * self.dim)
        self.transform_matrix = nn.Linear(self.dim, self.dim, bias=False)

        self.act = nn.LeakyReLU(negative_slope=self.leaky)

    def _parse_args(self, args):
        self.dim = int(args['embedding.size'])
        self.n_hop = int(args['n_hop'])
        self.kge_weight = float(args['kge_weight'])
        self.l2_weight = float(args['l2_weight'])
        self.n_memory = int(args['n_memory'])
        self.item_update_mode = args['item_update_mode']
        self.using_all_hops = True if args['using_all_hops'] == 'true' else False
        self.mode_pool = args['mode_pool']
        self.max_arity = int(args['max_arity'])
        self.dropout = float(args['dropout'])
        self.non_linear = args['non_linear']
        self.gnn_layers = int(args['gnn_layer'])
        self.temp = float(args['temp'])
        # self.keep_rate = float(args['keep_rate'])
        self.leaky = float(args['leaky'])

        conf = OptionConf(args['HKGRippleNet'])
        self.ss_rate = float(conf['-ss_rate'])

    def forward(self, kwargs, keep_rate=1):
        # [batch size, dim]
        items= kwargs['pos_items']
        memories_h = kwargs['memories_h']
        memories_r = kwargs['memories_r']
        memories_t = kwargs['memories_t']

        h_emb_list = []
        r_emb_list = []
        t_emb_list = []   

        uEmbed = self.uEmbed
        iEmbed = self.iEmbed

        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_emb_list.append(self.entity_emb(memories_h[i]))
            r_emb_list.append(
                self.relation_emb(memories_r[i]).view(
                    -1, self.n_memory, self.dim, self.dim
                )
            )
            # [batch size, n_memory, dim]
            t_emb_list.append(
                self.apply_non_linearity(self.entity_emb(memories_t[i]))
            )

        o_list, selected_item_embeddings = self._key_addressing(
            h_emb_list, r_emb_list, t_emb_list, self.item_embeddings.index_select(0, items)
        )

        # asssign new value to item_embeddings
        with torch.no_grad():
            self.iEmbed[items, :] = selected_item_embeddings
        return self.uEmbed, self.iEmbed, h_emb_list, t_emb_list, r_emb_list  

    def calculate_kg_loss(self, h_emb_list, t_emb_list, r_emb_list):
        kge_loss = 0    
        for hop in range(self.n_hop):
            # [batch size, n_memory, 1, dim]
            h_expanded = torch.unsqueeze(h_emb_list[hop], dim=2)
            # [batch size, n_memory, dim, 1]
            t_expanded = torch.unsqueeze(t_emb_list[hop], dim=3)
            # [batch size, n_memory, dim, dim]
            hRt = torch.squeeze(
                torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded)
            )
            kge_loss += torch.sigmoid(hRt).mean()
        kge_loss = self.kge_weight * kge_loss

        l2_loss = 0
        for hop in range(self.n_hop):
            l2_loss += (h_emb_list[hop] * h_emb_list[hop]).sum()
            l2_loss += (t_emb_list[hop] * t_emb_list[hop]).sum()
            l2_loss += (r_emb_list[hop] * r_emb_list[hop]).sum()
        l2_loss = self.l2_weight * l2_loss

        loss = kge_loss + l2_loss
        return loss 

    def _key_addressing(self, h_emb_list, r_emb_list, t_emb_list, item_embeddings):
        o_list = []

        for hop in range(self.n_hop):
            # [batch_size, n_memory, dim, 1]
            h_expanded = torch.unsqueeze(h_emb_list[hop], dim=3)

            # [batch_size, n_memory, dim]
            Rh = torch.squeeze(torch.matmul(r_emb_list[hop], h_expanded))

            # [batch_size, dim, 1]
            v = torch.unsqueeze(item_embeddings, dim=2) # bias 

            # [batch_size, n_memory]
            probs = torch.squeeze(torch.matmul(Rh, v))

            # [batch_size, n_memory]
            probs_normalized = F.softmax(probs, dim=1)

            # [batch_size, n_memory, 1]
            probs_expanded = torch.unsqueeze(probs_normalized, dim=2)

            # [batch_size, dim]
            o = (torch.squeeze(t_emb_list[hop], -1) * probs_expanded).sum(dim=1) # h * R * t 
            
            item_embeddings = self._update_item_embedding(item_embeddings, o)
            o_list.append(o)
        return o_list, item_embeddings

    def _update_item_embedding(self, item_embeddings, o):
        if self.item_update_mode == "replace":
            item_embeddings = o
        elif self.item_update_mode == "plus":
            item_embeddings = item_embeddings + o
        elif self.item_update_mode == "replace_transform":
            item_embeddings = self.transform_matrix(o)
        elif self.item_update_mode == "plus_transform":
            item_embeddings = self.transform_matrix(item_embeddings + o)
        else:
            raise Exception("Unknown item updating mode: " + self.item_update_mode)
        return item_embeddings
