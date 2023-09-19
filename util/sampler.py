from random import shuffle,randint,choice,sample
import numpy as np
from util.torch_interface import TorchGraphInterface
import torch
import random 

def next_batch_unified(data, data_kg, batch_size, batch_size_kg, n_negs=1, device=None):
    ptr = 0
    cf_data = np.array(data.training_data)
    kg_data = data_kg.kg_train_data.to_numpy()
    
    cf_size = len(cf_data)
    # data_size = len(kg_data)

    shuffle(kg_data)
    shuffle(cf_data)
    
    lst_user_item = list(set(list(cf_data[:,0]) + list(cf_data[:,1])))
    train_kg_dict = {k: data_kg.train_kg_dict[k] for k in lst_user_item}

    exist_heads= train_kg_dict.keys()
    
    h_list = list(exist_heads)
    h_dict = {value: idx for idx, value in enumerate(h_list)}
    
    all_tails = []
    pos_tail_sets = {}
    for head, tails in train_kg_dict.items():
        all_tails += [it[0] for it in tails] 
        pos_tail_sets[head] =  set([it[0] for it in tails])
    all_tails = list(set(all_tails))
    
    while ptr < cf_size:
        if ptr + batch_size < cf_size:
            batch_end = ptr + batch_size
        else:   
            batch_end = cf_size
            
        users = cf_data[ptr:batch_end, 0]
        items = cf_data[ptr:batch_end, 1]
        
        ptr = batch_end
        
        # select random items
        u_idx, i_idx, j_idx = [], [], []
        item_list = list(data.item.keys())
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            for m in range(n_negs):
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(data.item[neg_item])

        u_idx  = torch.LongTensor(u_idx).to(device)
        i_idx  = torch.LongTensor(i_idx).to(device)
        j_idx  = torch.LongTensor(j_idx).to(device)
        
        # selected entities 
        h_idx, r_idx, pos_t_idx, neg_t_idx = [], [], [], []
        selected_kg_data = []
        
        for i, h in enumerate(train_kg_dict.keys()):
            lst_values = train_kg_dict[h]
            for val in lst_values:
                selected_kg_data.append([int(h), val[1], val[0]])
        
        selected_kg_data = np.array(selected_kg_data)
        selected_indices  = np.random.randint(len(selected_kg_data), size=batch_size_kg)
        selected_kg_data = selected_kg_data[selected_indices, :]

        heads, relations, tails  = selected_kg_data[:,0], selected_kg_data[:,1], selected_kg_data[:,2]
        # time1 = datetime.datetime.now()
        
        h_idx.extend([h_dict[int(h)] for h in heads])
        r_idx.extend([int(rel) for rel in relations])
        pos_t_idx.extend([int(tail) for tail in tails])

        for head in heads:
            neg_t = random.choice(all_tails)
            while neg_t in pos_tail_sets[head]:
                neg_t = random.choice(all_tails)
            neg_t_idx.append(neg_t)

        h_idx  = torch.LongTensor(h_idx).to(device)
        r_idx  = torch.LongTensor(r_idx).to(device)
        pos_t_idx = torch.LongTensor(pos_t_idx).to(device)
        neg_t_idx  = torch.LongTensor(neg_t_idx).to(device)
        yield u_idx, i_idx, j_idx, h_idx, r_idx, pos_t_idx, neg_t_idx
        
def next_batch_unified_(data, data_kg, batch_size, batch_size_kg, n_negs=1, device=None):
    ptr = 0

    cf_data = data.training_data
    kg_data = data_kg.kg_train_data.to_numpy()
    
    cf_size = len(cf_data)
    data_size = len(kg_data)

    shuffle(kg_data)
    shuffle(cf_data)
    
    kg_dict = data_kg.train_kg_dict
    exist_heads= kg_dict.keys()
    h_list = list(exist_heads)
    h_dict = {value: idx for idx, value in enumerate(h_list)}
    all_tails = list(set(kg_data[:,2]))
    # Pre-compute positive tail sets and negative tails for each head
    pos_tail_sets = {head: set([it[0] for it in tails]) for head, tails in kg_dict.items()}
    
    while ptr < data_size:
        if ptr + batch_size_kg < data_size:
            batch_end = ptr + batch_size_kg
        else:   
            batch_end = data_size
        
        heads, relations, tails = kg_data[ptr:batch_end, 0], kg_data[ptr:batch_end, 1], kg_data[ptr:batch_end, 2]
        
        ptr = batch_end
        h_idx, r_idx, pos_t_idx, neg_t_idx = [], [], [], []
        # time1 = datetime.datetime.now()
        h_idx = [h_dict[head] for head in heads]
        
        r_idx.extend([int(rel) for rel in relations])
        pos_t_idx.extend([int(tail) for tail in tails])
        for head in heads:
            neg_t = random.choice(all_tails)
            while neg_t in pos_tail_sets[head]:
                neg_t = random.choice(all_tails)
            neg_t_idx.append(neg_t)
        # select random items
        selected_indices = np.random.choice(cf_size, batch_size)
        users = [cf_data[idx][0] for idx in selected_indices]
        items = [cf_data[idx][1] for idx in selected_indices]

        # select random items
        u_idx, i_idx, j_idx = [], [], []
        item_list = list(data.item.keys())
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            for m in range(n_negs):
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(data.item[neg_item])

        u_idx  = torch.LongTensor(u_idx).to(device)
        i_idx  = torch.LongTensor(i_idx).to(device)
        j_idx  = torch.LongTensor(j_idx).to(device)

        h_idx  = torch.LongTensor(h_idx).to(device)
        r_idx  = torch.LongTensor(r_idx).to(device)
        neg_t_idx  = torch.LongTensor(neg_t_idx).to(device)
        yield u_idx, i_idx, j_idx, h_idx, r_idx, pos_t_idx, neg_t_idx

def next_batch_kg(rec, batch_size):
    # sample data for KG
    def sample_pos_triples_for_h(kg_dict, head, n_sample_pos_triples):
        pos_triples = kg_dict[head]
        n_pos_triples = len(pos_triples)

        sample_relations, sample_pos_tails = [], []
        while True:
            if len(sample_relations) == n_sample_pos_triples:
                break

            pos_triple_idx = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
            
            tail = pos_triples[pos_triple_idx][0]
            relation = pos_triples[pos_triple_idx][1]

            if relation not in sample_relations and tail not in sample_pos_tails:
                sample_relations.append(relation)
                sample_pos_tails.append(tail)
        return sample_relations, sample_pos_tails

    def sample_neg_triples_for_h(kg_dict, head, relation, n_sample_neg_triples, highest_neg_idx):
        pos_triples = kg_dict[head]

        sample_neg_tails = []
        while True:
            if len(sample_neg_tails) == n_sample_neg_triples:
                break
            
            tail = np.random.randint(low=0, high=highest_neg_idx, size=1)[0]
            while tail not in kg_dict.keys():
                tail = np.random.randint(low=0, high=highest_neg_idx, size=1)[0]
            
            # xac nhan tail duoc chon k nam trong tap positive va chua tung duoc chon
            if (tail, relation) not in pos_triples and tail not in sample_neg_tails:
                sample_neg_tails.append(tail)
        return sample_neg_tails
    
    data_kg = rec.data_kg
    kg_dict = data_kg.train_kg_dict
    # ui_data = rec.data.training_data 
    
    exist_heads = list(kg_dict.keys())
    highest_neg_idx = data_kg.n_entities 
    ptr = 0 
    data_size = len(exist_heads)
    
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_head = random.sample(exist_heads, batch_size)
            batch_end = ptr + batch_size
        else:
            batch_head = [random.choice(exist_heads) for _ in range(batch_size)]
            batch_end = data_size
            
        ptr = batch_end

        batch_relation, batch_pos_tail, batch_neg_tail = [], [], []
        for h in batch_head:
            relation, pos_tail = sample_pos_triples_for_h(kg_dict, h, 1)
            batch_relation += relation
            batch_pos_tail += pos_tail
            
            neg_tail = sample_neg_triples_for_h(kg_dict, h, relation[0], 1, highest_neg_idx)
            batch_neg_tail += neg_tail
        
        batch_head = [data_kg.entity[i] for i in batch_head]
        batch_relation = [data_kg.relation[i] for i in batch_relation]
        batch_pos_tail = [data_kg.entity[i] for i in batch_pos_tail]
        batch_neg_tail = [data_kg.entity[i] for i in batch_neg_tail]
        
        batch_head = torch.LongTensor(batch_head).cuda()
        batch_relation = torch.LongTensor(batch_relation).cuda()
        batch_pos_tail = torch.LongTensor(batch_pos_tail).cuda()
        batch_neg_tail = torch.LongTensor(batch_neg_tail).cuda()

        yield batch_head, batch_relation, batch_pos_tail, batch_neg_tail    


def next_batch_pairwise(data,batch_size,n_negs=1, device=None):
    training_data = data.training_data
    shuffle(training_data)
    ptr = 0
    data_size = len(training_data)
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:   
            batch_end = data_size
        users = [training_data[idx][0] for idx in range(ptr, batch_end)]
        items = [training_data[idx][1] for idx in range(ptr, batch_end)]
        ptr = batch_end
        u_idx, i_idx, j_idx = [], [], []
        item_list = list(data.item.keys())
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            for m in range(n_negs):
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(data.item[neg_item])

        u_idx  = torch.LongTensor(u_idx).to(device)
        i_idx  = torch.LongTensor(i_idx).to(device)
        j_idx  = torch.LongTensor(j_idx).to(device)
        yield u_idx, i_idx, j_idx

def next_batch_pointwise(data,batch_size):
    training_data = data.training_data
    data_size = len(training_data)
    ptr = 0
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
        users = [training_data[idx][0] for idx in range(ptr, batch_end)]
        items = [training_data[idx][1] for idx in range(ptr, batch_end)]
        ptr = batch_end
        u_idx, i_idx, y = [], [], []
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            y.append(1)
            for instance in range(4):
                item_j = randint(0, data.item_num - 1)
                while data.id2item[item_j] in data.training_set_u[user]:
                    item_j = randint(0, data.item_num - 1)
                u_idx.append(data.user[user])
                i_idx.append(item_j)
                y.append(0)
        yield u_idx, i_idx, y

def next_batch_sequence(data, batch_size,n_negs=1,max_len=50):
    training_data = list(data.original_seq.values())
    shuffle(training_data)
    ptr = 0
    data_size = len(training_data)
    item_list = list(range(1,data.item_num+1))
    while ptr < data_size:
        if ptr+batch_size<data_size:
            batch_end = ptr+batch_size
        else:
            batch_end = data_size
        seq = np.zeros((batch_end-ptr, max_len),dtype=np.int)
        pos = np.zeros((batch_end-ptr, max_len),dtype=np.int)
        y =np.zeros((batch_end-ptr, max_len),dtype=np.int)
        neg = np.zeros((batch_end-ptr, max_len),dtype=np.int)
        seq_len = []
        for n in range(0, batch_end-ptr):
            start = len(training_data[ptr + n]) > max_len and -max_len or 0
            end =  len(training_data[ptr + n]) > max_len and max_len-1 or len(training_data[ptr + n])-1
            seq[n, :end] = training_data[ptr + n][start:-1]
            seq_len.append(end)
            pos[n, :end] = list(range(1,end+1))
            y[n, :end]=training_data[ptr + n][start+1:]
            negatives=sample(item_list,end)
            while len(set(negatives).intersection(set(training_data[ptr + n][start:-1]))) >0:
                negatives = sample(item_list, end)
            neg[n,:end]=negatives
        ptr=batch_end
        yield seq, pos, y, neg, np.array(seq_len,np.int)
