import numpy as np
import pandas as pd 
from data.loader import FileIO
import collections  
from collections import defaultdict
import torch
import scipy.sparse as sp
from data.graph import Graph
from data.ui_graph import Interaction

class Knowledge(Interaction, Graph):
    def __init__(self, conf, training, test, knowledge):
        super().__init__(conf, training, test)
        self.conf = conf 
        self.kg_data = knowledge

        self.entity = {}
        self.id2ent = {}

        self.userent = {}
        self.itement = {}
        
        self.u2id = {}
        self.id2u = {}
        
        self.i2id = {}
        self.id2i = {}
        
        self.relation = {}
        self.id2rel = {}

        self.cf_train_data = np.array(training)
        self.training_set_e = defaultdict(dict)

        self.construct_data()
        
        self.laplacian_type = 'random-walk'
        self.create_adjacency_dict()
        self.create_laplacian_dict()
        
        self.edge_index_kg, self.kg_interaction_mat = self.__create_sparse_knowledge_interaction_matrix()
        self.norm_kg_adj = self.normalize_graph_mat(self.kg_interaction_mat)

    def construct_data(self):
        kg_data = self.kg_data
        n_relations = max(kg_data['r']) + 1
        inverse_kg_data = kg_data.copy()
        inverse_kg_data = inverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
        inverse_kg_data['r'] += n_relations

        kg_data = pd.concat([kg_data, inverse_kg_data], axis=0, ignore_index=True, sort=False)

        # remap user_id 
        kg_data['r'] += 2
        
        kg_train_data = pd.concat([kg_data, inverse_kg_data], axis=0, ignore_index=True, sort=False)
        self.n_entities = max(max(kg_train_data['h']), max(kg_train_data['t'])) + 1

        # add interactions to kg data
        cf2kg_train_data = pd.DataFrame(np.zeros((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        cf2kg_train_data['h'] = self.cf_train_data[:,0]
        cf2kg_train_data['t'] = self.cf_train_data[:,1]

        inverse_cf2kg_train_data = pd.DataFrame(np.ones((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        inverse_cf2kg_train_data['h'] = self.cf_train_data[:,1]
        inverse_cf2kg_train_data['t'] = self.cf_train_data[:,0]

        self.kg_train_data = pd.concat([kg_train_data, cf2kg_train_data, inverse_cf2kg_train_data], ignore_index=True)
        self.n_kg_train = len(self.kg_train_data)

        self.n_users_entities = int(max(max(self.kg_train_data['h']), max(self.kg_train_data['t'])) + 1)
        self.n_relations = max(self.kg_train_data['r']) + 1

        # construct kg dict
        h_list = []
        t_list = []
        r_list = []

        self.train_kg_dict = collections.defaultdict(list)
        self.train_relation_dict = collections.defaultdict(list)

        for idx, row in self.kg_train_data.iterrows():
            h, r, t = int(row['h']), int(row['r']), int(row['t'])
            h_list.append(h)
            t_list.append(t)
            r_list.append(r)

            if h not in self.entity:
                self.entity[h] = len(self.entity)
                self.id2ent[self.entity[h]] = h
                # check h co phai user hay item k
                if h in self.user:
                    self.userent[h] = len(self.userent)
                #     # self.id2userent[self.userent[h]] = h
                if h in self.item:
                    self.itement[h] = len(self.itement)
                #     # self.id2itement[self.itement[h]] = h

            if t not in self.entity:
                self.entity[t] = len(self.entity)
                self.id2ent[self.entity[t]] = t 
                # check h co phai user hay item k 
                if t in self.user:
                    self.userent[t] = len(self.userent)
                #     # self.id2userent[self.userent[t]] = t
                if t in self.item:
                    self.itement[t] = len(self.itement)
                #     # self.id2itement[self.itement[t]] = t
            if r not in self.relation:
                self.relation[r] = len(self.relation)
                self.id2rel[self.relation[r]] = r 
            
            self.training_set_e[t][h] = r
            self.train_kg_dict[h].append((t, r))
            self.train_relation_dict[r].append((h, t))
        
        self.h_list = torch.LongTensor(h_list)
        self.t_list = torch.LongTensor(t_list)
        self.r_list = torch.LongTensor(r_list)
        
        lst_user_entities = list(self.userent.keys())
        lst_item_entities = list(self.itement.keys())

        for idx, u in enumerate(lst_user_entities):
            self.u2id[u] = idx
            self.id2u[idx] = u
        for idx, i in enumerate(lst_item_entities):
            self.i2id[i] = idx
            self.id2i[idx] = i
        
    def get_entity_id(self, e):
        if e in self.entity:
            return self.entity[e]
    
    def __create_sparse_knowledge_interaction_matrix(self):
        """
            return a sparse adjacency matrix with the shape (entity number, entity number)
        """
        row, col, entries = [], [], []
        for idx, pair in self.kg_train_data.iterrows():
            head, tail = int(pair['h']), int(pair['t'])
            row += [head]
            col += [tail]
            entries += [1.0]
            
        edge_index_kg = torch.LongTensor([row, col])
        interaction_mat = sp.csr_matrix((entries, (row, col)), shape=(self.n_users_entities, self.n_users_entities),dtype=np.float32)
        return edge_index_kg, interaction_mat
    
    def convert_coo2tensor(self, coo):
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))
    
    def create_adjacency_dict(self):
        self.adjacency_dict = {}
        for r, ht_list in self.train_relation_dict.items():
            rows = [e[0] for e in ht_list]
            cols = [e[1] for e in ht_list]
            vals = [1] * len(rows)
            adj = sp.coo_matrix((vals, (rows, cols)), shape=(self.n_users_entities, self.n_users_entities))
            self.adjacency_dict[r] = adj
    
    def create_laplacian_dict(self):
        def symmetric_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return norm_adj.tocoo()

        def random_walk_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))

            d_inv = np.power(rowsum, -1.0).flatten()
            d_inv[np.isinf(d_inv)] = 0
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        if self.laplacian_type == 'symmetric':
            norm_lap_func = symmetric_norm_lap
        elif self.laplacian_type == 'random-walk':
            norm_lap_func = random_walk_norm_lap
        
        self.laplacian_dict = {}
        for r, adj in self.adjacency_dict.items():
            self.laplacian_dict[r] = norm_lap_func(adj)

class KnowledgeNew(Interaction):
    def __init__(self, conf, training, test, knowledge):
        super().__init__(conf, training, test)
        self.conf = conf 
        self.kg_data = knowledge

        self.entity = {}
        self.id2ent = {}

        self.userent = {}
        self.itement = {}
        
        self.u2id = {}
        self.id2u = {}
        
        self.i2id = {}
        self.id2i = {}
        
        self.relation = {}
        self.id2rel = {}

        self.cf_train_data = np.array(training)
        self.training_set_e = defaultdict(dict)

        self.construct_data()
    
        self.laplacian_type = 'random-walk'
        self.create_adjacency_dict()
        self.create_laplacian_dict()
        
        self.edge_index_kg, self.kg_interaction_mat = self.__create_sparse_knowledge_interaction_matrix()
        self.interaction_mat = self.__create_sparse_interaction_matrix()
        
        
    def construct_data(self):
        kg_data = self.kg_data
        n_relations = max(kg_data['r']) + 1
        inverse_kg_data = kg_data.copy()
        inverse_kg_data = inverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
        inverse_kg_data['r'] += n_relations

        kg_data = pd.concat([kg_data, inverse_kg_data], axis=0, ignore_index=True, sort=False)

        # remap user_id 
        kg_data['r'] += 2
        
        kg_train_data = pd.concat([kg_data, inverse_kg_data], axis=0, ignore_index=True, sort=False)
        self.n_entities = max(max(kg_train_data['h']), max(kg_train_data['t'])) + 1
        self.n_relations = max(kg_train_data['r']) + 1

        # add interactions to kg data
        cf2kg_train_data = pd.DataFrame(np.zeros((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        cf2kg_train_data['h'] = self.cf_train_data[:,0]
        cf2kg_train_data['t'] = self.cf_train_data[:,1]

        inverse_cf2kg_train_data = pd.DataFrame(np.ones((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        inverse_cf2kg_train_data['h'] = self.cf_train_data[:,1]
        inverse_cf2kg_train_data['t'] = self.cf_train_data[:,0]

        self.kg_train_data = pd.concat([kg_train_data, cf2kg_train_data, inverse_cf2kg_train_data], ignore_index=True)
        self.n_kg_train = len(self.kg_train_data)

        self.n_users_entities = int(max(max(self.kg_train_data['h']), max(self.kg_train_data['t'])) + 1)

        # construct kg dict
        h_list = []
        t_list = []
        r_list = []

        self.train_kg_dict = collections.defaultdict(list)
        self.train_relation_dict = collections.defaultdict(list)

        for idx, row in self.kg_train_data.iterrows():
            h, r, t = int(row['h']), int(row['r']), int(row['t'])
            h_list.append(h)
            t_list.append(t)
            r_list.append(r)

            if h not in self.entity:
                self.entity[h] = len(self.entity)
                self.id2ent[self.entity[h]] = h
                # check h co phai user hay item k
                if h in self.user:
                    self.userent[h] = len(self.userent)
                #     # self.id2userent[self.userent[h]] = h
                if h in self.item:
                    self.itement[h] = len(self.itement)
                #     # self.id2itement[self.itement[h]] = h

            if t not in self.entity:
                self.entity[t] = len(self.entity)
                self.id2ent[self.entity[t]] = t 
                # check h co phai user hay item k 
                if t in self.user:
                    self.userent[t] = len(self.userent)
                #     # self.id2userent[self.userent[t]] = t
                if t in self.item:
                    self.itement[t] = len(self.itement)
                #     # self.id2itement[self.itement[t]] = t
            if r not in self.relation:
                self.relation[r] = len(self.relation)
                self.id2rel[self.relation[r]] = r 
            
            self.training_set_e[t][h] = r
            self.train_kg_dict[h].append((t, r))
            self.train_relation_dict[r].append((h, t))
        
        self.h_list = torch.LongTensor(h_list)
        self.t_list = torch.LongTensor(t_list)
        self.r_list = torch.LongTensor(r_list)
        
        lst_user_entities = list(self.userent.keys())
        lst_item_entities = list(self.itement.keys())

        for idx, u in enumerate(lst_user_entities):
            self.u2id[u] = idx
            self.id2u[idx] = u
        for idx, i in enumerate(lst_item_entities):
            self.i2id[i] = idx
            self.id2i[idx] = i
        
    def get_entity_id(self, e):
        if e in self.entity:
            return self.entity[e]
    
    def __create_sparse_knowledge_interaction_matrix(self):
        """
            return a sparse adjacency matrix with the shape (entity number, entity number)
        """
        row, col, entries = [], [], []
        for idx, pair in self.kg_train_data.iterrows():
            head, tail = int(pair['h']), int(pair['t'])
            row += [head]
            col += [tail]
            entries += [1.0]
        interaction_mat = sp.csr_matrix((entries, (row, col)), shape=(self.n_users_entities, self.n_users_entities),dtype=np.float32)
        return interaction_mat
    
    def __create_sparse_interaction_matrix(self):
        row, col, entries = [], [], []
        for pair in self.training_data:
            head, tail  = int(pair[0]), int(pair[1])
            row += [head]
            col += [tail]
            entries += [1.0]
        interaction_mat = sp.csr_matrix((entries, (row, col)), shape=(self.n_users_entities, self.n_users_entities),dtype=np.float32) 
        return interaction_mat
    
    def convert_coo2tensor(self, coo):
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))
    
    def create_adjacency_dict(self):
        self.adjacency_dict = {}
        for r, ht_list in self.train_relation_dict.items():
            rows = [e[0] for e in ht_list]
            cols = [e[1] for e in ht_list]
            vals = [1] * len(rows)
            adj = sp.coo_matrix((vals, (rows, cols)), shape=(self.n_users_entities, self.n_users_entities))
            self.adjacency_dict[r] = adj
    
    def create_laplacian_dict(self):
        def symmetric_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return norm_adj.tocoo()

        def random_walk_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))

            d_inv = np.power(rowsum, -1.0).flatten()
            d_inv[np.isinf(d_inv)] = 0
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        if self.laplacian_type == 'symmetric':
            norm_lap_func = symmetric_norm_lap
        elif self.laplacian_type == 'random-walk':
            norm_lap_func = random_walk_norm_lap
        
        self.laplacian_dict = {}
        for r, adj in self.adjacency_dict.items():
            self.laplacian_dict[r] = norm_lap_func(adj)
