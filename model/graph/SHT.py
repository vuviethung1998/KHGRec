import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os 
import numpy as np 

import time
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from util.loss_torch import bpr_loss, l2_reg_loss, EmbLoss, contrastLoss
from util.init import *
from base.torch_interface import TorchGraphInterface
from torch.optim.lr_scheduler import ReduceLROnPlateau

from util.evaluation import early_stopping

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SHT(GraphRecommender):
    def __init__(self, conf, training_set, test_set, knowledge_set, **kwargs):
        GraphRecommender.__init__(self, conf, training_set, test_set, knowledge_set, **kwargs)
        self.reg_loss = EmbLoss() 
        self.kwargs = kwargs 
        self.model = SHTEncoder(self.data, kwargs)
        self._parse_config(self.config, kwargs)
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lRate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=self.lr_decay, patience=5)
        
    def _parse_config(self, config, kwargs):
        self.maxEpoch = int(kwargs['max_epoch'])
        self.batchSize = int(kwargs['batch_size'])
        self.lRate = float(kwargs['lrate'])
        self.lr_decay = float(kwargs['lr_decay'])
        self.maxEpoch = int(kwargs['max_epoch'])
        self.batchSize = int(kwargs['batch_size'])
        self.reg = float(kwargs['reg'])
        self.latent_size = int(kwargs['embedding_size'])
        self.embeddingSize = int(kwargs['hyper_dim'])
        self.hyperDim = int(kwargs['hyper_dim'])
        self.dropRate = float(kwargs['drop_rate'])
        self.negSlove = float(kwargs['p'])
        self.nLayers = int(kwargs['n_layers'])
        self.ss_rate = float(kwargs['cl_rate'])
        self.temp = float(kwargs['temp'])
        self.seed = int(kwargs['seed'])
        self.early_stopping_steps = int(kwargs['early_stopping_steps'])
        
    def calcLosses(self, ancs, poss, negs, gcnEmbedsLst, hyperEmbedsLst, reg):
        bprLoss = bpr_loss(ancs, poss, negs)
        sslLoss = 0
        for i in range(self.nLayers):
            embeds1 = gcnEmbedsLst[i].detach()
            embeds2 = hyperEmbedsLst[i]
            sslLoss += contrastLoss(embeds1[:self.data.n_users], embeds2[:self.data.n_users], torch.unique(ancs.long()), self.temp)\
                         + contrastLoss(embeds1[self.data.n_users:], embeds2[self.data.n_users:], torch.unique(poss.long()), self.temp)
        sslLoss *= self.ss_rate
        return bprLoss, sslLoss

    def train(self, load_pretrained=False):
        model = self.model 
        recall_list = []
        total_train_losses = []
        total_rec_losses = []
        total_reg_losses = []
        lst_performances = []


        if load_pretrained:
            model.load_state_dict(torch.load(f"./results/{self.kwargs['model']}/{self.kwargs['dataset']}/model_full/{self.kwargs['model']}-weight.pth"))
            model.eval()
            with torch.no_grad():
                _, self.user_emb, self.item_emb = model()
                cur_data, data_ep = self.fast_evaluation(0, train_time=0)
                lst_performances.append(data_ep)
        else:
            for ep in range(self.maxEpoch):
                train_losses = []
                rec_losses = []
                reg_losses = []
                s_train = time.time()
                
                for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                    user_idx, pos_idx, neg_idx = batch                
                    model.train()
                    bprLoss, sslLoss = model.calcLosses(user_idx, pos_idx, neg_idx)
                    regLoss = model.calcRegLoss(model) * self.reg
                    loss = bprLoss + regLoss + sslLoss
                    train_losses.append(loss.item())
                    self.optimizer.zero_grad()
                    loss.backward()                 
                    self.optimizer.step()
                
                e_train = time.time() 
                tr_time = e_train - s_train 

                batch_train_loss = np.mean(train_losses)
                batch_rec_loss = np.mean(rec_losses)
                batch_reg_loss = np.mean(reg_losses)
                
                total_train_losses.append([ep, batch_train_loss])
                total_rec_losses.append([ep, batch_rec_loss])
                total_reg_losses.append([ep, batch_reg_loss])
                self.scheduler.step(batch_train_loss)
                
                # Evaluation
                model.eval()
                with torch.no_grad():
                    _, self.user_emb, self.item_emb = model()
                    cur_data, data_ep = self.fast_evaluation(ep, train_time=tr_time)
                    lst_performances.append(data_ep)
                    
                    cur_recall =  float(cur_data[2].split(':')[1])
                    recall_list.append(cur_recall)
                    best_recall, should_stop = early_stopping(recall_list, self.early_stopping_steps)
                    if should_stop:
                        break 
            # return hyperUEmbeds, hyperIEmbeds
        self.save_perfomance_training(lst_performances)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            _, self.best_user_emb, self.best_item_emb = self.model()
            print("Saving")
            self.save_model(self.model)     
            
    def predict(self, u):
        user_id  = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[user_id], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class SHTEncoder(nn.Module):
    def __init__(self, data, args):
        super(SHTEncoder, self).__init__()
        
        init = nn.init.xavier_uniform_

        self.args = args
        self.data = data
        self.n_user = self.data.n_users
        self.n_item = self.data.n_items
        self._parse_config(args)
        # adj
        self.norm_adj = self.data.norm_adj
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).to(device)

        self.uEmbeds = nn.Parameter(init(torch.empty(self.n_user, self.embeddingSize)).to(device))
        self.iEmbeds = nn.Parameter(init(torch.empty(self.n_item , self.embeddingSize)).to(device))
        self.uHyper = nn.Parameter(init(torch.empty(args['hyperedge_num'], self.embeddingSize)).to(device))
        self.iHyper = nn.Parameter(init(torch.empty(args['hyperedge_num'], self.embeddingSize)).to(device))
        
    def _parse_config(self, kwargs):
        self.maxEpoch = int(kwargs['max_epoch'])
        self.batchSize = int(kwargs['batch_size'])
        self.lRate = float(kwargs['lrate'])
        self.lr_decay = float(kwargs['lr_decay'])
        self.maxEpoch = int(kwargs['max_epoch'])
        self.batchSize = int(kwargs['batch_size'])
        self.reg = float(kwargs['reg'])
        self.latent_size = int(kwargs['embedding_size'])
        self.embeddingSize = int(kwargs['hyper_dim'])
        self.hyperDim = int(kwargs['hyper_dim'])
        self.dropRate = float(kwargs['drop_rate'])
        self.negSlove = float(kwargs['p'])
        self.nLayers = int(kwargs['n_layers'])
        self.ss_rate = float(kwargs['cl_rate'])
        self.temp = float(kwargs['temp'])
        self.seed = int(kwargs['seed'])
        self.edgeSampRate = 0.1 
        self.ssl1_reg = 0.1
        self.ssl2_reg = 0.1
        self.early_stopping_steps = int(kwargs['early_stopping_steps'])
        self.hyperedge_num = int(kwargs['hyperedge_num'])
        
    def gcnLayer(self, adj, embeds):
        return torch.spmm(adj, embeds)

    def hgnnLayer(self, embeds, hyper):
    # HGNN can also be seen as learning a transformation in hidden space, with args.hyperNum hidden units (hyperedges)
        return embeds @ (hyper.T @ hyper)# @ (embeds.T @ embeds)

    def forward(self):
        adj = self.sparse_norm_adj
        embeds = torch.concat([self.uEmbeds, self.iEmbeds], dim=0)
        lats = [embeds]
        for i in range(self.nLayers):
            temlat = self.gcnLayer(adj, lats[-1])
            lats.append(temlat)
        embeds = sum(lats)
        # this detach helps eliminate the mutual influence between the local GCN and the global HGNN
        hyperUEmbeds = self.hgnnLayer(embeds[:self.n_user].detach(), self.uHyper)
        hyperIEmbeds = self.hgnnLayer(embeds[self.n_user:].detach(), self.iHyper)
        return embeds, hyperUEmbeds, hyperIEmbeds

    def pickEdges(self, adj):
        idx = adj._indices()
        rows, cols = idx[0, :], idx[1, :]
        mask = torch.logical_and(rows <= self.n_user, cols > self.n_user)
        rows, cols = rows[mask], cols[mask]
        edgeSampNum = int(self.edgeSampRate * rows.shape[0])
        if edgeSampNum % 2 == 1:
            edgeSampNum += 1
        edgeids = torch.randint(rows.shape[0], [edgeSampNum])
        pckUsrs, pckItms = rows[edgeids], cols[edgeids] - self.n_user
        return pckUsrs, pckItms

    def pickRandomEdges(self, adj):
        edgeNum = adj._indices().shape[1]
        edgeSampNum = int(self.edgeSampRate * edgeNum)
        if edgeSampNum % 2 == 1:
            edgeSampNum += 1
        rows = torch.randint(self.n_user, [edgeSampNum])
        cols = torch.randint(self.n_item, [edgeSampNum])
        return rows, cols

    def bprLoss(self, uEmbeds, iEmbeds, ancs, poss, negs):
        ancEmbeds = uEmbeds[ancs]
        posEmbeds = iEmbeds[poss]
        negEmbeds = iEmbeds[negs]
        scoreDiff = self.pairPredict(ancEmbeds, posEmbeds, negEmbeds)
        bprLoss = - ((scoreDiff).sigmoid() + 1e-6).log().mean()
        return bprLoss
    
    def calcRegLoss(self, model):
        ret = 0
        for W in model.parameters():
            ret += W.norm(2).square()
        # ret += (model.usrStruct + model.itmStruct)
        return ret

    def calcLosses(self, ancs, poss, negs):
        adj = self.sparse_norm_adj
        embeds, hyperU, hyperI = self.forward()
        uEmbeds, iEmbeds = embeds[:self.n_user], embeds[self.n_user:]

        bprLoss = self.bprLoss(uEmbeds, iEmbeds, ancs, poss, negs) + self.bprLoss(hyperU, hyperI, ancs, poss, negs)
        # the sample generation can be further generalized as this
        pckUsrs, pckItms = self.pickRandomEdges(adj)
        # we can simply apply a symmetric manner for prediction align, without the cumbersome meta network
        _scores1 = (hyperU[pckUsrs] * hyperI[pckItms]).sum(-1)
        _scores2 = (uEmbeds[pckUsrs] * iEmbeds[pckItms]).sum(-1)
        halfNum = _scores1.shape[0] // 2
        fstScores1 = _scores1[:halfNum]
        scdScores1 = _scores1[halfNum:]
        fstScores2 = _scores2[:halfNum]
        scdScores2 = _scores2[halfNum:]
        scores1 = ((fstScores1 - scdScores1) / self.temp).sigmoid()
        scores2 = ((fstScores2 - scdScores2) / self.temp).sigmoid()
        # prediction alignment in a BPR-like scheme
        sslLoss1 = -(scores2.detach() * (scores1 + 1e-8).log() + (1 - scores2.detach()) * (1 - scores1 + 1e-8).log()).mean() * self.ss_rate
        sslLoss2 = -(scores1.detach() * (scores2 + 1e-8).log() + (1 - scores1.detach()) * (1 - scores2 + 1e-8).log()).mean() * self.ss_rate
        sslLoss = sslLoss1 + sslLoss2
        return bprLoss, sslLoss

    def predict(self):
        embeds, hyperU, hyperI = self.forward()
        return hyperU, hyperI

    def pairPredict(self, ancEmbeds, posEmbeds, negEmbeds):
        return self.innerProduct(ancEmbeds, posEmbeds) - self.innerProduct(ancEmbeds, negEmbeds)
    
    def innerProduct(self, usrEmbeds, itmEmbeds):
        return torch.sum(usrEmbeds * itmEmbeds, dim=-1)