import torch
import random
from random import shuffle, choice

def next_batch_pairwise(data, batch_size, n_negs=1, device=None):
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
            i_idx.append(items[i])
            u_idx.append(user)
            for m in range(n_negs):
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(neg_item)

        u_idx  = torch.LongTensor(u_idx).to(device)
        i_idx  = torch.LongTensor(i_idx).to(device)
        j_idx  = torch.LongTensor(j_idx).to(device)
        yield u_idx, i_idx, j_idx
    
def next_batch_kg(data_kg, batch_size, n_negs=1, device=None):
    kg_data = data_kg.kg_train_data.to_numpy()
    kg_dict = data_kg.train_kg_dict
    shuffle(kg_data)

    ptr = 0
    exist_heads= kg_dict.keys()
    h_list = list(exist_heads)
    h_dict = {value: idx for idx, value in enumerate(h_list)}
    all_tails = list(set(kg_data[:,2]))
    data_size = len(kg_data)
    pos_tail_sets = {head: set([it[0] for it in tails]) for head, tails in kg_dict.items()}
    
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:   
            batch_end = data_size
        heads, relations, tails = kg_data[ptr:batch_end, 0], kg_data[ptr:batch_end, 1], kg_data[ptr:batch_end, 2]
        ptr = batch_end
        h_idx, r_idx, pos_t_idx, neg_t_idx = [], [], [], []
        # time1 = datetime.datetime.now()
        h_idx = [h_dict[head] for head in heads]
        r_idx = [int(rel) for rel in relations]
        pos_t_idx = [int(pos_t) for pos_t in tails]

        for head in heads:
            neg_t = random.choice(all_tails)
            while neg_t in pos_tail_sets[head]:
                neg_t = random.choice(all_tails)
            try:
                neg_t_idx.append(int(h_dict[neg_t]))
            except:
                neg_t_idx.append(1234)        
        h_idx  = torch.LongTensor(h_idx).to(device)
        r_idx  = torch.LongTensor(r_idx).to(device)
        pos_t_idx  = torch.LongTensor(pos_t_idx).to(device)
        neg_t_idx  = torch.LongTensor(neg_t_idx).to(device)
        yield h_idx, r_idx, pos_t_idx, neg_t_idx
            