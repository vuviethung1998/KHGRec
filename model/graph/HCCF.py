import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise_kg, next_batch_pairwise
from util.loss_torch import bpr_loss, l2_reg_loss, EmbLoss, contrastLoss
from util.init import *
from base.torch_interface import TorchGraphInterface

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HCCF(GraphRecommender):
    def __init__(self, conf, training_set, test_set, **kwargs):
        GraphRecommender.__init__(self, conf, training_set, test_set, **kwargs)
        # config = OptionConf(self.config['HGNN'])

        self.reg_loss = EmbLoss() 
        self.model = HCCFModel(self.config, self.data )

        self._parse_config(self.config)
        self.model.to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), weight_decay=1e-3, lr=self.lRate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=self.lr_decay,patience=7)

    def _parse_config(self, config):
        self.lRate = float(config['learnRate'])
        self.lr_decay = float(config['learnRateDecay'])
        self.maxEpoch = int(config['num.max.epoch'])
        self.batchSize = int(config['batch_size'])
        self.reg = float(config['reg.lambda'])
        self.embeddingSize = int(config['embedding.size'])
        self.hyperDim = int(config['hyper.size'])
        self.dropRate = float(config['dropout'])
        self.negSlove = float(config['leaky'])
        self.nLayers = int(config['gnn_layer'])

    def train(self):
        model = self.model 

        for ep in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batchSize)):
                user_idx, pos_idx, neg_idx = batch
                model.train()
                # s_model = time.time()
                rec_user_emb, rec_item_emb, _, _ = model(keep_rate=1- self.dropRate)
                bpr_loss, ssl_loss = model.calculate_loss(user_idx, pos_idx, neg_idx, keep_rate= 1-self.dropRate)
                batch_loss = bpr_loss + ssl_loss 

                self.optimizer.zero_grad()
                batch_loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
                self.optimizer.step()
            model.eval()
            with torch.no_grad():
                self.user_emb, self.item_emb, _, _ = model(keep_rate=1)
            
            s_eval = time.time()
            self.fast_evaluation(ep)
            e_eval = time.time()
            print("Eval time: %f s" % (e_eval - s_eval))

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb, _,_ = self.model(keep_rate=1)

    def predict(self, u):
        user_id  = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[user_id], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class HCCFModel(nn.Module):
    def __init__(self, config, data):
        super(HCCFModel, self).__init__()
        self.data = data
        self._parse_args(config)

        init = nn.init.xavier_uniform_
        adj = self.data.ui_adj
        self.adj  = TorchGraphInterface.convert_sparse_mat_to_tensor(adj).to(device) 
        
        # init embedding
        self.user_embedding = nn.Parameter(init(torch.zeros(self.data.user_num, self.input_dim)))
        self.item_embedding = nn.Parameter(init(torch.zeros(self.data.item_num, self.input_dim)))

        self.uHyper = nn.Parameter(init(torch.zeros(self.input_dim, self.hyper_dim)))
        self.iHyper = nn.Parameter(init(torch.zeros(self.input_dim, self.hyper_dim)))

        self.edgeDropper = SpAdjDropEdge()
        self.gcnLayer = GCNLayer(self.leaky)
        self.hgnnLayer = HGNNLayer(self.leaky, self.hyper_dim)
        
    def _parse_args(self, config):
        self.gnn_layer = int(config['gnn_layer'])
        self.input_dim = int(config['embedding.size'])
        self.hyper_dim = int(config['hyper.size'])
        self.drop_rate = float(config['dropout'])
        self.leaky = float(config['leaky'])
        self.temp = float(config['temp'])

    def calculate_loss(self, ancs, poss, negs, keep_rate):
        uEmbeds, iEmbeds, gcnEmbedsLst, hyperEmbedsLst = self.forward(keep_rate)

        ancEmbeds = uEmbeds[ancs]
        posEmbeds = iEmbeds[poss]
        negEmbeds = iEmbeds[negs]
        scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
        bprLoss = - (scoreDiff).sigmoid().log().mean()

        sslLoss = 0
        for i in range(self.gnn_layer):
            embeds1 = gcnEmbedsLst[i].detach()
            embeds2 = hyperEmbedsLst[i]
            sslLoss += contrastLoss(embeds1[:self.data.user_num], embeds2[:self.data.user_num], torch.unique(ancs), self.temp) + contrastLoss(embeds1[self.data.user_num:], embeds2[self.data.user_num:], torch.unique(poss), self.temp)
        return bprLoss, sslLoss
        
    def forward(self, keep_rate):
        uEmbed = self.user_embedding   
        iEmbed = self.item_embedding      
        
        # print("uEmbed: ")
        # print(uEmbed)  
        embeds = torch.cat((uEmbed, iEmbed), dim=0)
        lats = [embeds]
        gnnLats = []
        hyperLats = []
        uuHyper = uEmbed @ self.uHyper
        iiHyper = iEmbed @ self.iHyper

        for i in range(self.gnn_layer):
            
            dropped_edge = self.edgeDropper(self.adj, keep_rate)
            temEmbeds = self.gcnLayer(dropped_edge.to_dense(), lats[-1])
            hyperULat = self.hgnnLayer(F.dropout(uuHyper, p=1-keep_rate), lats[-1][:self.data.user_num])
            hyperILat = self.hgnnLayer(F.dropout(iiHyper, p=1-keep_rate), lats[-1][self.data.user_num:])
            gnnLats.append(temEmbeds)
            hyperLats.append(torch.cat([hyperULat, hyperILat], dim=0))
            lats.append(temEmbeds + hyperLats[-1])
        embeds = sum(lats)
        user_embed =  embeds[:self.data.user_num]
        item_embed = embeds[self.data.user_num:]
        # print(user_embed)
        # print(item_embed)
        return user_embed, item_embed, gnnLats, hyperLats

class GCNLayer(nn.Module):
	def __init__(self, leaky):
		super(GCNLayer, self).__init__()
		self.act = nn.LeakyReLU(negative_slope=leaky)

	def forward(self, adj, embeds):
		return self.act(torch.spmm(adj, embeds))

class HGNNLayer(nn.Module):
    def __init__(self, leaky,  hyper_dim):
        super(HGNNLayer, self).__init__()
        self.hyper_dim = hyper_dim
        self.act = nn.LeakyReLU(negative_slope=leaky)
        self.fc1 = nn.Linear(hyper_dim, hyper_dim ,bias=False) 
        self.fc2 = nn.Linear(hyper_dim, hyper_dim ,bias=False)  
        self.fc3 = nn.Linear(hyper_dim, hyper_dim ,bias=False)  

    def forward(self, adj, embeds):
        
        lat1 = self.act(adj.T @ embeds)
        lat2 = self.act(self.fc1(lat1.T).T) +  lat1
        lat3 = self.act(self.fc2(lat2.T).T) + lat2
        lat4 = self.act(self.fc3(lat3.T).T) + lat3 
        ret = self.act(adj @ lat4)
        return ret

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

def innerProduct(usrEmbeds, itmEmbeds):
	return torch.sum(usrEmbeds * itmEmbeds, dim=-1)

def pairPredict(ancEmbeds, posEmbeds, negEmbeds):
	return innerProduct(ancEmbeds, posEmbeds) - innerProduct(ancEmbeds, negEmbeds)

def calcRegLoss(model):
	ret = 0
	for W in model.parameters():
		ret += W.norm(2).square()
	# ret += (model.usrStruct + model.itmStruct)
	return ret

def contrastLoss(embeds1, embeds2, nodes, temp):
	embeds1 = F.normalize(embeds1 + 1e-8, p=2)
	embeds2 = F.normalize(embeds2 + 1e-8, p=2)
	pckEmbeds1 = embeds1[nodes]
	pckEmbeds2 = embeds2[nodes]
	nume = torch.exp(torch.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
	deno = torch.exp(pckEmbeds1 @ pckEmbeds2.T / temp).sum(-1) + 1e-8
	return -torch.log(nume / deno).mean()



