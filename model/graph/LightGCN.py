import torch
import torch.nn as nn
import numpy as np 
import pandas as pd 
import time
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss,l2_reg_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util.evaluation import early_stopping

# paper: LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR'20\
class LightGCN(GraphRecommender):
    def __init__(self, conf, training_set, test_set, knowledge_set, **kwargs):
        super(LightGCN, self).__init__(conf, training_set, test_set, knowledge_set, **kwargs)
        args = OptionConf(self.config['LightGCN'])
        self.kwargs = kwargs
        self.n_layers = int(args['-n_layer'])
        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers)
        self.lr_decay  = float(kwargs['lr_decay'])
        self.early_stopping_steps = int(kwargs['early_stopping_steps'])
        self.reg = float(kwargs['reg'])
        self.maxEpoch = int(kwargs['max_epoch'])
        self.lRate = float(kwargs['lrate'])
        self.wdecay = float(kwargs['weight_decay'])
        
    def train(self, load_pretrained):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate, weight_decay=self.wdecay)

        lst_train_losses = []
        lst_rec_losses = []
        lst_reg_losses = []
        lst_performances = []
        recall_list = []


        for epoch in range(self.maxEpoch):
            train_losses = []
            rec_losses = []
            reg_losses = []
            s_train = time.time()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) 
                reg_loss =  l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb) /self.batch_size
                batch_loss = rec_loss + reg_loss

                train_losses.append(batch_loss.item())
                rec_losses.append(rec_loss.item())
                reg_losses.append(reg_loss.item())

                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            e_train = time.time() 
            tr_time = e_train - s_train 
            train_loss = np.mean(train_losses)
            rec_loss = np.mean(rec_losses)
            reg_loss = np.mean(reg_losses)

            lst_train_losses.append([epoch, train_loss])
            lst_rec_losses.append([epoch,rec_loss])
            lst_reg_losses.append([epoch, reg_loss])

            with torch.no_grad():
                self.user_emb, self.item_emb = model()
                cur_data, data_ep = self.fast_evaluation(epoch, train_time=tr_time)
                lst_performances.append(data_ep)
                
                cur_recall =  float(cur_data[2].split(':')[1])
                recall_list.append(cur_recall)
                best_recall, should_stop = early_stopping(recall_list, self.early_stopping_steps)
                if should_stop:
                    break 
        
        print(lst_train_losses)
        self.save_loss(lst_train_losses, lst_rec_losses, lst_reg_losses)
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

class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.n_users, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.n_items, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.data.n_users]
        item_all_embeddings = all_embeddings[self.data.n_users:]
        return user_all_embeddings, item_all_embeddings


