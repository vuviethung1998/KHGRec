import torch
import torch.nn as nn
import torch.nn.functional as F
import time 
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from data.augmentor import GraphAugmentor
from util.evaluation import early_stopping

class SGL(GraphRecommender):
    def __init__(self, conf, training_set, test_set, knowledge_set, **kwargs):
        super(SGL, self).__init__(conf, training_set, test_set, knowledge_set, **kwargs)
        self._parse_config(self.config, kwargs)
        
        self.model = SGL_Encoder(self.data, self.emb_size, self.drop_rate, self.nLayers, self.temp, self.aug_type)
        self.lr_decay  = float(kwargs['lr_decay'])
        self.weight_decay = float(kwargs['weight_decay'])
        self.early_stopping_steps = int(kwargs['early_stopping_steps'])
        self.reg = float(kwargs['reg'])
        self.embeddingSize = int(kwargs['hyper_dim'])
    
    def _parse_config(self, config, kwargs):        
        self.maxEpoch = int(kwargs['max_epoch'])
        self.batchSize = int(kwargs['batch_size'])
        
        self.lRate = float(kwargs['lrate'])
        self.lr_decay = float(kwargs['lr_decay'])
        self.maxEpoch = int(kwargs['max_epoch'])
        self.batchSize = int(kwargs['batch_size'])
        self.reg = float(kwargs['reg'])
        self.latent_size = int(kwargs['embedding_size'])
        self.hyperDim = int(kwargs['hyper_dim'])
        self.drop_rate = float(kwargs['drop_rate'])
        self.leaky = float(kwargs['p'])
        self.nLayers = int(kwargs['n_layers'])
        self.cl_rate = float(kwargs['cl_rate'])
        
        self.temp = float(kwargs['temp'])
        self.seed = int(kwargs['seed'])
        self.early_stopping_steps = int(kwargs['early_stopping_steps'])
        self.aug_type = 0
        
    def train(self, load_pretrained):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate, weight_decay=self.weight_decay)
        
        lst_performances = []
        recall_list = []
        
        for epoch in range(self.maxEpoch):
            dropped_adj1 = model.graph_reconstruction()
            dropped_adj2 = model.graph_reconstruction()

            s_train = time.time() 

            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.cl_rate * model.cal_cl_loss([user_idx,pos_idx],dropped_adj1,dropped_adj2)
                batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb,neg_item_emb) + cl_loss
                # Backward and optimize

                optimizer.zero_grad()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 4)
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            
            e_train = time.time() 
            tr_time = e_train - s_train 


            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
                cur_data, data_ep =  self.fast_evaluation(epoch, train_time=tr_time)
                
                cur_recall =  float(cur_data[2].split(':')[1])
                recall_list.append(cur_recall)
                best_recall, should_stop = early_stopping(recall_list, self.early_stopping_steps)
                if should_stop:
                    break 
            lst_performances.append(data_ep)

        self.save_perfomance_training(lst_performances)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()
            print("Saving")
            self.save_model(self.model)  
            
    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class SGL_Encoder(nn.Module):
    def __init__(self, data, emb_size, drop_rate, n_layers, temp, aug_type):
        super(SGL_Encoder, self).__init__()
        self.data = data
        self.drop_rate = drop_rate
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.temp = temp
        self.aug_type = aug_type
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.n_users, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.n_items, self.emb_size))),
        })
        return embedding_dict

    def graph_reconstruction(self):
        if self.aug_type==0 or 1:
            dropped_adj = self.random_graph_augment()
        else:
            dropped_adj = []
            for k in range(self.n_layers):
                dropped_adj.append(self.random_graph_augment())
        return dropped_adj

    def random_graph_augment(self):
        dropped_mat = None
        if self.aug_type == 0:
            dropped_mat = GraphAugmentor.node_dropout(self.data.interaction_mat, self.drop_rate)
        elif self.aug_type == 1 or self.aug_type == 2:
            dropped_mat = GraphAugmentor.edge_dropout(self.data.interaction_mat, self.drop_rate)
        dropped_mat = self.data.convert_to_laplacian_mat(dropped_mat)
        return TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat).cuda()

    def forward(self, perturbed_adj=None):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers):

            if perturbed_adj is not None:
                if isinstance(perturbed_adj,list):
                    ego_embeddings = torch.sparse.mm(perturbed_adj[k], ego_embeddings)
                else:
                    ego_embeddings = torch.sparse.mm(perturbed_adj, ego_embeddings)
            else:
                ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.n_users, self.data.n_items])
        return user_all_embeddings, item_all_embeddings

    def cal_cl_loss(self, idx, perturbed_mat1, perturbed_mat2):
        if type(idx[0]) is not list:
            u_idx = torch.unique(idx[0])
        else:
            u_idx = torch.unique(torch.Tensor(idx[0]).cuda().type(torch.long))
        if type(idx[1]) is not list:
            i_idx = torch.unique(idx[1])
        else:
            i_idx = torch.unique(torch.Tensor(idx[1]).cuda().type(torch.long))
        user_view_1, item_view_1 = self.forward(perturbed_mat1)
        user_view_2, item_view_2 = self.forward(perturbed_mat2)
        view1 = torch.cat((user_view_1[u_idx],item_view_1[i_idx]),0)
        view2 = torch.cat((user_view_2[u_idx],item_view_2[i_idx]),0)
        return InfoNCE(view1,view2,self.temp)
