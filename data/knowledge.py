import numpy as np
import pandas as pd 
from data.loader import FileIO
import collections  
from collections import defaultdict
import torch
import scipy.sparse as sp

from data.ui_graph import Interaction 

class Knowledge(Interaction):
    def __init__(self, conf,  training, test, knowledge):
        super(Knowledge, self).__init__(conf,  training, test, knowledge)
        self.conf = conf    

        self.entity = {}
        self.id2ent = {}
        self.training_set_e = defaultdict(dict)

        self.construct_data(knowledge)

        self.n_entities = len(self.training_set_e)
        self.kg_interaction_mat = self.__create_sparse_knowledge_interaction_matrix()

    def construct_data(self, kg_data):
        n_relations = max(kg_data['r']) + 1
        inverse_kg_data = kg_data.copy()
        inverse_kg_data = inverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
        inverse_kg_data['r'] += n_relations
        self.kg_train_data = pd.concat([kg_data, inverse_kg_data], axis=0, ignore_index=True, sort=False)

        # self.kg_train_data = pd.concat([kg_data, cf2kg_train_data, inverse_cf2kg_train_data], ignore_index=True)
        self.n_kg_train = len(self.kg_train_data)

        self.n_users_entities = self.n_users + self.n_entities

        self.cf_train_data = (np.array(list(map(lambda d: d + self.n_entities, self.cf_train_data[0]))).astype(np.int32), self.cf_train_data[1].astype(np.int32))
        self.cf_test_data = (np.array(list(map(lambda d: d + self.n_entities, self.cf_test_data[0]))).astype(np.int32), self.cf_test_data[1].astype(np.int32))

        self.train_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.train_user_dict.items()}
        self.test_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.test_user_dict.items()}

        # add interactions to kg data
        cf2kg_train_data = pd.DataFrame(np.zeros((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        cf2kg_train_data['h'] = self.cf_train_data[0]
        cf2kg_train_data['t'] = self.cf_train_data[1]

        inverse_cf2kg_train_data = pd.DataFrame(np.ones((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        inverse_cf2kg_train_data['h'] = self.cf_train_data[1]
        inverse_cf2kg_train_data['t'] = self.cf_train_data[0]

        self.kg_train_data = pd.concat([kg_data, cf2kg_train_data, inverse_cf2kg_train_data], ignore_index=True)

        # construct kg dict
        h_list = []
        t_list = []
        r_list = []

        self.train_kg_dict = collections.defaultdict(list)
        self.train_relation_dict = collections.defaultdict(list)

        for row in self.kg_train_data.iterrows():
            h, r, t = row
            h_list.append(h)
            t_list.append(t)
            r_list.append(r)

            if t not in self.entity:
                self.entity[t] = len(self.entity)
                self.id2ent[self.entity[t]] = t 
            self.training_set_e[t][h] = r

            self.train_kg_dict[h].append((t, r))
            self.train_relation_dict[r].append((h, t))

        self.h_list = torch.LongTensor(h_list)
        self.t_list = torch.LongTensor(t_list)
        self.r_list = torch.LongTensor(r_list)

    def get_entity_id(self, e):
        if e in self.entity:
            return self.entity[e]

    def __create_sparse_knowledge_interaction_matrix(self):
        """
            return a sparse adjacency matrix with the shape (entity number, entity number)
        """
        row, col, entries = [], [], []
        for pair in self.kg_train_data.iterrows():
            row += [self.entity[pair[0]]]
            col += [self.entity[pair[2]]]
            entries += [1.0]

        interaction_mat = sp.csr_matrix((entries, (row, col)), shape=(self.entity_num, self.entity_num),dtype=np.float32)
        return interaction_mat

# class Knowledge(object):
#     def __init__(self, conf, knowledge_set,item_dict, testing_set_i, item_num):
#         # self.config = config
#         print("Loading the dataset {} ....".format(conf['dataset']))
#         self.entity_num, self.relation_num, kg_data = knowledge_set
#         self.kg = self.construct_kg(kg_data)
#         # import pdb; pdb.set_trace()
#         self.train_item_set  = set(list(item_dict.keys()))
#         self.test_item_set  = testing_set_i
#         self.training_knowledge_data  =  []
#         self.item = item_dict
#         self.entity = {}
#         self.id2ent = {}
#         self.training_set_e = defaultdict(dict)
#         self.training_set_ie = defaultdict(dict)
#         self.test_set_entity = defaultdict(dict)
#         self.__generate_set()

#         # print(self.training_set_e)
#         self.entity_num = len(self.training_set_e)
#         self.item_num = item_num

#         self.kg_interaction_mat = self.__create_sparse_knowledge_interaction_matrix()
#         self.kg_ui_adj = self.__create_sparse_knowledge_bipartite_adjacency()

#     def construct_kg(self, kg_np):
#         print('constructing knowledge graph ...')
#         kg = collections.defaultdict(list)
#         for head, relation, tail in kg_np:
#             head, relation, tail = int(head),int(relation), int(tail)
#             kg[head].append((tail, relation))
#         return kg

#     def __generate_set(self):
#         for it in list(self.train_item_set):
#             for ent in self.kg[it]:
#                 tail, rel = ent
#                 if tail not in self.entity:
#                     self.entity[tail] = len(self.entity)
#                     self.id2ent[self.entity[tail]] = tail 
#                 self.training_set_ie[it][tail] = rel 
#                 self.training_set_e[tail][it] = rel 
#                 self.training_knowledge_data.append([it, tail, rel])

#         for it in list(self.test_item_set):
#             for ent in self.kg[it]:
#                 tail, rel = ent
#                 if  tail not in self.entity:
#                     continue 
#                 self.test_set_entity[it][tail] = rel

#     def get_entity_id(self, e):
#         if e in self.entity:
#             return self.entity[e]

#     def __create_sparse_knowledge_interaction_matrix(self):
#         """
#             return a sparse adjacency matrix with the shape (user number, item number)
#         """
#         row, col, entries = [], [], []
#         for pair in self.training_knowledge_data:
#             row += [self.item[pair[0]]]
#             col += [self.entity[pair[1]]]
#             entries += [1.0]

#         interaction_mat = sp.csr_matrix((entries, (row, col)), shape=(self.item_num,self.entity_num),dtype=np.float32)
#         return interaction_mat

#     def __create_sparse_knowledge_bipartite_adjacency(self, self_connection=False): 
#         '''
#         return a sparse adjacency matrix with the shape (user number + item number, user number + item number)
#         '''
#         n_nodes = self.item_num + self.entity_num 
#         row_idx = [self.item[pair[0]] for pair in self.training_knowledge_data]
#         col_idx = [self.entity[pair[1]] for pair in self.training_knowledge_data]
#         item_np = np.array(row_idx)
#         entity_np = np.array(col_idx)
#         ratings = np.ones_like(item_np, dtype=np.float32)
#         tmp_adj = sp.csr_matrix((ratings, (item_np, entity_np + self.item_num)), shape=(n_nodes, n_nodes),dtype=np.float32)
#         adj_mat = tmp_adj + tmp_adj.T
#         if self_connection:
#             adj_mat += sp.eye(n_nodes)
#         return adj_mat

