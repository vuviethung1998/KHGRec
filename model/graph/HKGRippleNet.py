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

class HKGRippleNet(GraphRecommender):
    def __init__(self, conf, training_set, test_set, **kwargs):
        GraphRecommender.__init__(self, conf, training_set, test_set, **kwargs)
        
        # self.n_layers = int(args['-n_layer'])
        # self.ss_rate = float(args['-ss_rate'])
        self.reg = float(self.config['reg.lambda'])
        
        self.kwargs = kwargs
        # self.social_data = Relation(conf, kwargs['social.data'], self.data.user)
        self.reg_loss = EmbLoss()
        # self.model = HKGRippleNetModel(self.config, self.data)
        self.model = HKGRippleNetKGModel(self.config, self.data)
    
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
            # ---------------------------------------------------------------
            
            # ----------------------------hccf + ripple ------------------------
            # for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
            #     user_idx, pos_idx, neg_idx = batch
                # ---------------------------------------------------------------


                # --------------------hccf + ripple----------------------------
                model.train()
                user_emb, item_emb, h_emb_list, t_emb_list, r_emb_list = model(kwargs, keep_rate=float(self.config['keep_rate']))
                # ------------------------------------------------
                # ---------------------hccf ---------------------------
                # user_emb, item_emb = model(keep_rate=float(self.config['keep_rate']))

                ancEmbds = user_emb[user_idx]
                posEmbds = item_emb[pos_idx]
                negEmbds = item_emb[neg_idx]

                # print("User_emb: \n")
                # print(ancEmbds)
                # print("Pos_item_emb: \n")
                # print(posEmbds)
                # print("Neg_item_emb: \n")
                # print(negEmbds)
                # ---------------------------------------------------------

                # --------------------hccf + ripple----------------------------
                # batch_loss = rec_loss + reg_loss + ssl_loss
                rec_loss,  reg_loss, kge_loss, ssl_loss = self.calculate_loss(ancEmbds, posEmbds, negEmbds, h_emb_list, t_emb_list, r_emb_list,user_idx, pos_idx)
                batch_loss = rec_loss + reg_loss + ssl_loss + kge_loss 

                # print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(),'reg_loss', reg_loss.item(), 'kge_loss', kge_loss.item(), 'ssl_loss', ssl_loss.item() )
                # --------------------------------------------------------------
                
                # -------------------hccf -----------------------------------
                # rec_loss, reg_loss, ssl_loss =  self.calculate_loss_hccf(ancEmbds, posEmbds, negEmbds, user_idx, pos_idx)

                # batch_loss = rec_loss + reg_loss + ssl_loss
                # print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(),'reg_loss', reg_loss.item() ,'ssl_loss', ssl_loss.item() )

                # ----------------------------------------------------
                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 4)
                optimizer.step()

                train_losses.append(batch_loss.item())

                # --------------------hccf + ripple----------------------------
                # if n % 100 == 0 and n > 0:
                #     print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(),'reg_loss', reg_loss.item(), 'kge_loss', kge_loss.item() )
                # --------------------------------------------------------------
            
            batch_train_loss = np.mean(train_losses)
            scheduler.step(batch_train_loss)

            model.eval()
            with torch.no_grad():
                # --------------------hccf + ripple----------------------------
                self.user_emb, self.item_emb, _, _, _ = model(kwargs,keep_rate=1)
                # --------------------------------------------------------------

                #----------------------hccf--------------------------------------
                # self.user_emb, self.item_emb = model(keep_rate=1)
                # --------------------------------------------------------------
            
            s_eval = time.time()

            # --------------------hccf + ripple----------------------------
            self.fast_evaluation(epoch, kwargs)
            #--------------------------------------------
            #---------------------hccf -----------------------------------------
            # self.fast_evaluation(epoch)
            #--------------------------------------------
            e_eval = time.time()
            print("Eval time: %f s" % (e_eval - s_eval))
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def calculate_loss(self, user_emb, pos_item_emb, neg_item_emb, h_emb_list, t_emb_list, r_emb_list, user_idx, pos_idx):
        reg_loss = self.reg * self.reg_loss(user_emb, pos_item_emb, neg_item_emb)
        rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
        ssl_loss = self.model.calculate_ssl_loss(user_idx, pos_idx)
        # ssl_loss = sslLoss * self.ss_rate 
        kge_loss = self.model.calculate_kg_loss(h_emb_list, t_emb_list, r_emb_list)
        return rec_loss, reg_loss, kge_loss, ssl_loss
    
    def calculate_loss_hccf(self, anc_emb, pos_emb, neg_emb, user_idx, pos_idx):
        bprLoss = bpr_loss(anc_emb, pos_emb, neg_emb)
        reg_loss = self.reg * self.reg_loss(anc_emb, pos_emb, neg_emb)

        sslLoss = self.model.calculate_ssl_loss(user_idx, pos_idx)
        ssl_loss = sslLoss * self.ss_rate 
        return bprLoss, reg_loss, ssl_loss 

    def save(self, kwargs=None):
        with torch.no_grad():
            # --------------------hccf + ripple----------------------------
            self.best_user_emb, self.best_item_emb, _, _, _ = self.model(kwargs, keep_rate=1)
            #------------------------------------------------------------
            #------------------------hccf------------------------------
            # self.best_user_emb, self.best_item_emb = self.model(keep_rate=1)
            #------------------------------------------------------------

    def predict(self, u):
        user_id  = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[user_id], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()  

class FC(nn.Module):
    def __init__(self, inputDim, outDim, leaky=0.1):
        super(FC,self).__init__()
        initializer = nn.init.xavier_normal_
        self.W_fc = nn.Parameter(initializer(torch.empty(inputDim, outDim).cuda()))
        # self.act = nn.LeakyReLU(negative_slope=leaky)
        self.act = nn.ReLU()

    def forward(self, inp):
        #W = self.W_fc.weight
        ret = self.act(inp @ self.W_fc)
        return ret

class HGNNLayer(nn.Module):
    def __init__(self,inputdim, hyperNum, leaky=0.1):
        super(HGNNLayer, self).__init__()
        self.inputdim = inputdim
        self.fc1 = FC(self.inputdim,hyperNum).cuda()
        self.fc2 = FC(self.inputdim,hyperNum).cuda()
        self.fc3 = FC(self.inputdim,hyperNum).cuda()
        self.actFunc = nn.LeakyReLU(leaky) 

    def forward(self,lats,adj):
        lat1 = self.actFunc(torch.transpose(adj,0,1) @ lats) #shape adj:user,hyperNum lats:user,latdim lat1:hypernum,latdim
        lat2 = torch.transpose(self.fc1(torch.transpose(lat1,0,1)),0,1) + lat1 #shape hypernum,latdim
        lat3 = torch.transpose(self.fc2(torch.transpose(lat2,0,1)),0,1) + lat2
        lat4 = torch.transpose(self.fc3(torch.transpose(lat3,0,1)),0,1) + lat3
        ret = adj @ lat4
        ret = self.actFunc(ret)
        return ret

class GCNLayer(nn.Module):
    def __init__(self, dim, leaky=0.5):
        super(GCNLayer, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=leaky)
        self.fc = FC(dim,dim)

    def forward(self, lats, adj):
        ret = self.act(self.fc(adj @ lats) )
        return ret 

class weight_trans(nn.Module):
    def __init__(self, dim):
        super(weight_trans, self).__init__()
        initializer = nn.init.xavier_normal_
        self.W = nn.Parameter(initializer(torch.empty(dim, dim).cuda()))

    def forward(self,normalize):
        ret = normalize @ self.W
        return ret

class HKGRippleNetKGModel(nn.Module):
    def __init__(self, args, data):
        super(HKGRippleNetKGModel, self).__init__()
        initializer = nn.init.xavier_normal_

        self.data = data
        self._parse_args(args)

        self.uEmbed0 = nn.Parameter(initializer(torch.empty(self.data.user_num, self.dim).to(device)))
        self.iEmbed0 = nn.Parameter(initializer(torch.empty(self.data.item_num, self.dim).to(device)))
        self.uhyper = nn.Parameter(initializer(torch.empty(self.dim, self.hyper_num).to(device)))
        self.ihyper = nn.Parameter(initializer(torch.empty(self.dim, self.hyper_num).to(device)))

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

        for i in range(self.gnn_layers):
            self.hyperULat_layers.append(HGNNLayer(self.hyper_num, self.hyper_num, self.leaky)) #shape hyperNum,hyperNum
            self.hyperILat_layers.append(HGNNLayer(self.hyper_num, self.hyper_num, self.leaky)) #shape hyperNum,hyperNum
            self.uLat_layers.append(GCNLayer(self.dim, self.leaky))
            self.iLat_layers.append(GCNLayer(self.dim, self.leaky))
            self.weight_layers.append(weight_trans(self.dim))

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
        self.hyper_num = int(args['hyper_num'])
        self.gnn_layers = int(args['gnn_layer'])
        self.temp = float(args['temp'])
        # self.keep_rate = float(args['keep_rate'])
        self.leaky = float(args['leaky'])

        conf = OptionConf(args['HKGRippleNet'])
        self.ss_rate = float(conf['-ss_rate'])

    def calcSSL(self, hyperLat, gnnLat):
        posScore = torch.exp(torch.sum(hyperLat * gnnLat, dim = 1) / self.temp)
        negScore = torch.sum(torch.exp(gnnLat @ torch.transpose(hyperLat, 0, 1) / self.temp), dim = 1)
        uLoss = torch.sum(-torch.log(posScore / (negScore + 1e-8) + 1e-8))
        return uLoss

    def edgeDropout(self, mat, drop):
        def dropOneMat(mat):
            indices = mat._indices().cpu()
            values = mat._values().cpu()
            shape = mat.shape
            newVals = nn.functional.dropout(values, p = drop)
            return torch.sparse.FloatTensor(indices, newVals, shape).to(torch.float32).cuda()
        return dropOneMat(mat)

    # def forward(self, kwargs, keep_rate=1):
    def forward(self, kwargs, keep_rate=1):
        # [batch size, dim]
        items= kwargs['pos_items']
        memories_h = kwargs['memories_h']
        memories_r = kwargs['memories_r']
        memories_t = kwargs['memories_t']

        h_emb_list = []
        r_emb_list = []
        t_emb_list = []   

        uEmbed0 = self.uEmbed0
        iEmbed0 = self.iEmbed0
        uhyper = self.uhyper
        ihyper = self.ihyper

        uuHyper = uEmbed0 @ uhyper#shape user,hyperNum
        iiHyper = iEmbed0 @ ihyper#shape item,hyperNum

        ulats = [uEmbed0]
        ilats = [iEmbed0]

        gnnULats = []
        gnnILats = []
        hyperULats = []
        hyperILats = []

        for i in range(self.gnn_layers):
            hyperULat = self.hyperULat_layers[i](ulats[-1],nn.functional.dropout(uuHyper, p = 1-keep_rate))
            hyperILat = self.hyperILat_layers[i](ilats[-1],nn.functional.dropout(iiHyper, p = 1-keep_rate))

            ulat = self.uLat_layers[i](ilats[-1], self.edgeDropout(self.adj_mat, drop = 1-keep_rate))
            ilat = self.iLat_layers[i](ulats[-1], self.edgeDropout(self.tp_adj_mat, drop = 1-keep_rate))

            gnnULats.append(ulat)
            gnnILats.append(ilat)

            hyperULats.append(hyperULat)
            hyperILats.append(hyperILat)

            ulats.append(ulat + hyperULat + ulats[-1])
            ilats.append(ilat + hyperILat + ilats[-1])

        self.user_embeddings = sum(ulats)
        self.item_embeddings = sum(ilats) 

        self.gnnUlats = gnnULats
        self.gnnIlats = gnnILats 
        self.hyperUlats = hyperULats
        self.hyperIlats = hyperILats 

        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_emb_list.append(self.entity_emb(memories_h[i]))
            r_emb_list.append(
                self.relation_emb(memories_r[i]).view(
                    -1, self.n_memory, self.dim, self.dim
                )
            )
            # [batch size, n_memory, dim]
            t_embs = []
            for j in range(self.max_arity):
                t_embs.append(self.entity_emb(memories_t[i][j]))

            if self.mode_pool == 'sum':
                t_emb = sum(t_embs)
            elif self.mode_pool  == 'average':
                t_emb = sum(t_embs) /  self.max_arity 
            elif self.mode_pool == 'pool':
                t_emb_stack  = torch.stack(t_embs, dim=-1)
                t_emb = torch.squeeze(self.pool_layer(t_emb_stack), -1)
                t_emb = self.apply_non_linearity(self.non_linear, t_emb)
            elif self.mode_pool == 'attention':
                # self-attention
                t_emb_att = self.attention_layer(**t_embs)
                t_emb = self.apply_non_linearity(self.non_linear,t_emb_att)
            elif self.mode_pool == 'reale':
                t_emb_stack = self.input_dropout(torch.stack(t_embs, dim=-1))

                t_embs = torch.squeeze(self.pool_layer(t_emb_stack), -1)
                t_emb = self.apply_non_linearity(self.non_linear, t_embs)
            t_emb_list.append(t_emb)

        o_list, selected_item_embeddings = self._key_addressing(
            h_emb_list, r_emb_list, t_emb_list, self.item_embeddings.index_select(0, items)
        )

        # asssign new value to item_embeddings
        with torch.no_grad():
            self.item_embeddings[items, :] = selected_item_embeddings
        return self.user_embeddings, self.item_embeddings, h_emb_list, t_emb_list, r_emb_list  

        # return uLat, iLat

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

    def calculate_ssl_loss(self, uids, iids):
        sslLoss = 0 
        uniqUids = torch.unique(uids)
        uniqIids = torch.unique(iids)

        for i in range(len(self.hyperUlats)):
            pckHyperULat = self.weight_layers[i](torch.nn.functional.normalize(torch.index_select(self.hyperUlats[i], 0, uniqUids), p=2, dim=1))# @ self.weight_layers[i].weight
            pckGnnULat = torch.nn.functional.normalize(torch.index_select(self.gnnUlats[i], 0, uniqUids), p=2, dim=1)
            pckhyperILat = self.weight_layers[i](torch.nn.functional.normalize(torch.index_select(self.hyperIlats[i], 0, uniqIids), p=2, dim=1))# @ self.weight_layers[i].weight
            pckGnnILat = torch.nn.functional.normalize(torch.index_select(self.gnnIlats[i], 0, uniqIids), p=2, dim=1)
            uLoss = self.calcSSL(pckHyperULat, pckGnnULat)
            iLoss = self.calcSSL(pckhyperILat, pckGnnILat)
            sslLoss += uLoss + iLoss
        sslLoss *= self.ss_rate
        return sslLoss

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

class HKGRippleNetModel(nn.Module):
    def __init__(self, args, data):
        super(HKGRippleNetModel, self).__init__()
        initializer = nn.init.xavier_normal_

        self.data = data
        self._parse_args(args)

        self.uEmbed0 = nn.Parameter(initializer(torch.empty(self.data.user_num, self.dim).to(device)))
        self.iEmbed0 = nn.Parameter(initializer(torch.empty(self.data.item_num, self.dim).to(device)))
        self.uhyper = nn.Parameter(initializer(torch.empty(self.dim, self.hyper_num).to(device)))
        self.ihyper = nn.Parameter(initializer(torch.empty(self.dim, self.hyper_num).to(device)))

        self.adj_mat = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.interaction_mat).to(device)
        self.tp_adj_mat = torch.transpose(self.adj_mat, 0,1)

        self.hyperULat_layers = nn.ModuleList()
        self.hyperILat_layers = nn.ModuleList()
        self.weight_layers = nn.ModuleList()

        self.uLat_layers = nn.ModuleList()
        self.iLat_layers = nn.ModuleList()
        
        for i in range(self.gnn_layers):
            self.hyperULat_layers.append(HGNNLayer(self.hyper_num, self.hyper_num, self.leaky)) #shape hyperNum,hyperNum
            self.hyperILat_layers.append(HGNNLayer(self.hyper_num, self.hyper_num, self.leaky)) #shape hyperNum,hyperNum
            self.uLat_layers.append(GCNLayer(self.dim, self.leaky))
            self.iLat_layers.append(GCNLayer(self.dim, self.leaky))
            self.weight_layers.append(weight_trans(self.dim))

        self.act = nn.LeakyReLU(negative_slope=self.leaky)
        # self.act = nn.ReLU()

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
        self.hyper_num = int(args['hyper_num'])
        self.gnn_layers = int(args['gnn_layer'])
        self.temp = float(args['temp'])
        # self.keep_rate = float(args['keep_rate'])
        self.leaky = float(args['leaky'])

    def calcSSL(self, hyperLat, gnnLat):
        posScore = torch.exp(torch.sum(hyperLat * gnnLat, dim = 1) / self.temp)
        negScore = torch.sum(torch.exp(gnnLat @ torch.transpose(hyperLat, 0, 1) / self.temp), dim = 1)
        uLoss = torch.sum(-torch.log(posScore / (negScore + 1e-8) + 1e-8))
        return uLoss

    # def Regularize(self, reg):
    #     ret = 0.0
    #     for i in range(len(reg)):
    #         ret += torch.sum(torch.square(reg[i]))
    #     return ret

    def edgeDropout(self, mat, drop):
        def dropOneMat(mat):
            indices = mat._indices().cpu()
            values = mat._values().cpu()
            shape = mat.shape
            newVals = nn.functional.dropout(values, p = drop)
            return torch.sparse.FloatTensor(indices, newVals, shape).to(torch.float32).cuda()
        return dropOneMat(mat)

    def forward(self, keep_rate=1):
        uEmbed0 = self.uEmbed0
        iEmbed0 = self.iEmbed0
        uhyper = self.uhyper
        ihyper = self.ihyper

        uuHyper = uEmbed0 @ uhyper#shape user,hyperNum
        iiHyper = iEmbed0 @ ihyper#shape item,hyperNum

        ulats = [uEmbed0]
        ilats = [iEmbed0]

        gnnULats = []
        gnnILats = []
        hyperULats = []
        hyperILats = []

        for i in range(self.gnn_layers):
            hyperULat = self.hyperULat_layers[i](ulats[-1],nn.functional.dropout(uuHyper, p = 1-keep_rate))
            hyperILat = self.hyperILat_layers[i](ilats[-1],nn.functional.dropout(iiHyper, p = 1-keep_rate))

            ulat = self.uLat_layers[i](ilats[-1], self.edgeDropout(self.adj_mat, drop = 1-keep_rate))
            ilat = self.iLat_layers[i](ulats[-1], self.edgeDropout(self.tp_adj_mat, drop = 1-keep_rate))

            gnnULats.append(ulat)
            gnnILats.append(ilat)

            hyperULats.append(hyperULat)
            hyperILats.append(hyperILat)

            ulats.append(ulat + hyperULat + ulats[-1])
            ilats.append(ilat + hyperILat + ilats[-1])
            # ulats.append(ulat + hyperULat )
            # ilats.append(ilat + hyperILat )

        uLat = sum(ulats)
        iLat = sum(ilats) 

        self.gnnUlats = gnnULats
        self.gnnIlats = gnnILats 
        self.hyperUlats = hyperULats
        self.hyperIlats = hyperILats 

        return uLat, iLat

    def calculate_ssl_loss(self, uids, iids):
        sslLoss = 0 
        uniqUids = torch.unique(uids)
        uniqIids = torch.unique(iids)

        for i in range(len(self.hyperUlats)):
            pckHyperULat = self.weight_layers[i](torch.nn.functional.normalize(torch.index_select(self.hyperUlats[i], 0, uniqUids), p=2, dim=1))# @ self.weight_layers[i].weight
            pckGnnULat = torch.nn.functional.normalize(torch.index_select(self.gnnUlats[i], 0, uniqUids), p=2, dim=1)
            pckhyperILat = self.weight_layers[i](torch.nn.functional.normalize(torch.index_select(self.hyperIlats[i], 0, uniqIids), p=2, dim=1))# @ self.weight_layers[i].weight
            pckGnnILat = torch.nn.functional.normalize(torch.index_select(self.gnnIlats[i], 0, uniqIids), p=2, dim=1)
            uLoss = self.calcSSL(pckHyperULat, pckGnnULat)
            iLoss = self.calcSSL(pckhyperILat, pckGnnILat)
            sslLoss += uLoss + iLoss
        return sslLoss
