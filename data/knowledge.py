import numpy as np
from data.loader import FileIO
import collections  
from collections import defaultdict

import scipy.sparse as sp

class Knowledge(object):
    def __init__(self, conf, knowledge_set,item_dict, testing_set_i, item_num):
        # self.config = config
        print("Loading the dataset {} ....".format(conf['dataset']))
        self.entity_num, self.relation_num, kg_data = knowledge_set
        self.kg = self.construct_kg(kg_data)
        self.train_item_set  = set(list(item_dict.keys()))
        self.test_item_set  = testing_set_i
        self.training_knowledge_data  =  []
        self.item = item_dict
        self.entity = {}
        self.id2ent = {}
        self.training_set_e = defaultdict(dict)
        self.training_set_i = defaultdict(dict)
        self.test_set_entity = defaultdict(dict)
        self.__generate_set()

        # print(self.training_set_e)
        self.entity_num = len(self.training_set_e)
        self.item_num = item_num

        self.kg_interaction_mat = self.__create_sparse_knowledge_interaction_matrix()
        self.kg_ui_adj = self.__create_sparse_knowledge_bipartite_adjacency()
        # print(self.kg_interaction_mat.shape)

    def construct_kg(self, kg_np):
        print('constructing knowledge graph ...')
        kg = collections.defaultdict(list)
        for head, relation, tail in kg_np:
            head, relation, tail = int(head),int(relation), int(tail)
            kg[head].append((tail, relation))
        return kg

    def __generate_set(self):
        for it in list(self.train_item_set):
            for ent in self.kg[it]:
                tail, rel = ent
                if tail not in self.entity:
                    self.entity[tail] = len(self.entity)
                    self.id2ent[self.entity[tail]] = tail
        
                self.training_set_i[it][tail] = rel 
                self.training_set_e[tail][it] = rel 
                self.training_knowledge_data.append((it, tail))

        for it in list(self.test_item_set):
            for ent in self.kg[it]:
                tail, rel = ent
                if  tail not in self.entity:
                    continue 
                self.test_set_entity[it][tail] = rel

    def get_entity_id(self, e):
        if e in self.entity:
            return self.entity[e]

    def __create_sparse_knowledge_interaction_matrix(self):
        """
            return a sparse adjacency matrix with the shape (user number, item number)
        """
        row, col, entries = [], [], []
        for pair in self.training_knowledge_data:
            row += [self.item[pair[0]]]
            col += [self.entity[pair[1]]]
            entries += [1.0]

        interaction_mat = sp.csr_matrix((entries, (row, col)), shape=(self.item_num,self.entity_num),dtype=np.float32)
        return interaction_mat

    def __create_sparse_knowledge_bipartite_adjacency(self, self_connection=False): 
        '''
        return a sparse adjacency matrix with the shape (user number + item number, user number + item number)
        '''
        n_nodes = self.item_num + self.entity_num 
        row_idx = [self.item[pair[0]] for pair in self.training_knowledge_data]
        col_idx = [self.entity[pair[1]] for pair in self.training_knowledge_data]
        item_np = np.array(row_idx)
        entity_np = np.array(col_idx)
        ratings = np.ones_like(item_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (item_np, entity_np + self.item_num)), shape=(n_nodes, n_nodes),dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T
        if self_connection:
            adj_mat += sp.eye(n_nodes)
        return adj_mat

# class Knowledge:
#     def __init__(self, config, mode):
#         # self.config = config
#         self.mode= mode 
#         self.batch_size = config['kg_batchsize']
#         self.neg_ratio = config['neg_ratio']
        
#         self.batch_index = 0
#         self.ent2id = {"":0}
#         self.rel2id = {"":0}
#         print("Loading the dataset {} ....".format(config['ds_name']))
#         self.n_entities, self.n_relations, self.kg_data, self.khg_data, self.max_arity = FileIO.load_kg_data(self.config['knowledge.data'])
#         # shuffle dataset
#         np.random.shuffle(self.kg_data)
#         np.random.shuffle(self.khg_data)
#         # if hypergraph or graph 
#         # self.max_arity = config['max_arity'] # 8
        
#     def num_rel(self):
#         return self.n_relations 
    
#     def num_ent(self):
#         return self.n_entities
    
#     def tuple2ids(self, tuple_):
#         output = np.zeros(self.max_arity + 1)
#         for ind,t in enumerate(tuple_):
#             if ind == 0:
#                 output[ind] = self.get_rel_id(t)
#             else:
#                 output[ind] = self.get_ent_id(t)
#         return output
    
#     def get_ent_id(self, ent):
#         if not ent in self.ent2id:
#             self.ent2id[ent] = len(self.ent2id)
#         return self.ent2id[ent]
    
#     def get_rel_id(self, rel):
#         if not rel in self.rel2id:
#             self.rel2id[rel] = len(self.rel2id)
#         return self.rel2id[rel]
    
#     def next_pos_batch(self):
#         batch_size = self.batch_size
#         if self.batch_index + batch_size < len(self.kg_data):
#             batch = self.kg_data[self.batch_index: self.batch_index+batch_size]
#             self.batch_index += batch_size
#         else:
#             batch = self.kg_data[self.batch_index:]
#             ###shuffle##
#             np.random.shuffle(self.kg_data)
#             self.batch_index = 0
#         batch = np.append(batch, np.zeros((len(batch), 1)), axis=1).astype("int") #appending the +1 label
#         batch = np.append(batch, np.zeros((len(batch), 1)), axis=1).astype("int") #appending the 0 arity
#         return batch
    
#     def next_batch(self):
#         mode = self.mode 
        
#         if mode == 'kg':
#             pos_batch = self.next_pos_batch()
#             batch = self.generate_neg(pos_batch, self.neg_ratio)
#             arities = batch[:,4]
#             ms = np.zeros((len(batch),3))
#             bs = np.ones((len(batch), 3))
#             for i in range(len(batch)):
#                 ms[i][0:arities[i]] = 1
#                 bs[i][0:arities[i]] = 0
#             r = torch.tensor(batch[:,1]).long().to(self.device)
#             e1 = torch.tensor(batch[:,0]).long().to(self.device)
#             e2 = torch.tensor(batch[:,2]).long().to(self.device)
#             labels = batch[:,3]
#             return r, e1, e2          
        
#         elif mode == 'khg':
#             pos_batch = self.next_pos_batch()
#             batch = self.generate_neg(pos_batch, self.neg_ratio)
             
#             arities = batch[:,10]
#             ms = np.zeros((len(batch),self.max_arity))
#             bs = np.ones((len(batch), 9))
#             for i in range(len(batch)):
#                 ms[i][0:arities[i]] = 1
#                 bs[i][0:arities[i]] = 0
#             r  = torch.tensor(batch[:,1]).long().to(self.device)
#             e1 = torch.tensor(batch[:,0]).long().to(self.device)
#             e2 = torch.tensor(batch[:,2]).long().to(self.device)
#             e3 = torch.tensor(batch[:,3]).long().to(self.device)
#             e4 = torch.tensor(batch[:,4]).long().to(self.device)
#             e5 = torch.tensor(batch[:,5]).long().to(self.device)
#             e6 = torch.tensor(batch[:,6]).long().to(self.device)
#             e7 = torch.tensor(batch[:,7]).long().to(self.device)
#             e8 = torch.tensor(batch[:,8]).long().to(self.device)
#             e9= torch.tensor(batch[:,9]).long().to(self.device)
#             labels = batch[:,10]
            
#             ms = torch.tensor(ms).float().to(self.device)
#             bs = torch.tensor(bs).float().to(self.device)
#             return r, e1, e2, e3, e4, e5, e6, e7, e8, e9, labels, ms, bs
            

#     def generate_neg(self, pos_batch, neg_ratio): # chua hieu lam
#         mode = self.mode 
#         if mode == 'kg': 
#             arities = [8 - (t == 0).sum() for t in pos_batch]
#             pos_batch[:,-1] = arities
#             neg_batch = np.concatenate([self.neg_each(np.repeat([c], neg_ratio * arities[i] + 1, axis=0), arities[i], neg_ratio) for i, c in enumerate(pos_batch)], axis=0)
#             return neg_batch
#         elif mode == 'khg':
#             pass
        
#     def neg_each(self, arr, arity, nr): # chua hieu lam
#         arr[0,-2] = 1
#         for a in range(arity):
#             arr[a* nr + 1:(a + 1) * nr + 1, a + 1] = np.random.randint(low=1, high=self.num_ent(), size=nr)
#         return arr

#     def was_last_batch(self):
#         return (self.batch_index == 0)

#     def num_batch(self, batch_size):
#         return int(math.ceil(float(len(self.data["train"])) / batch_size))
    
