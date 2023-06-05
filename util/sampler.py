from random import shuffle,randint,choice,sample
import numpy as np
from util.torch_interface import TorchGraphInterface
import torch

def next_batch_pairwise_kg(args, data, batch_size, n_negs=1):
    interaction_data = data

    interaction_mat = interaction_data.normalize_graph_mat(interaction_data.interaction_mat)
    interaction_mat = TorchGraphInterface.convert_sparse_mat_to_tensor(interaction_mat).cuda()
    
    training_data = data.training_data
    shuffle(training_data)
    
    ptr= 0
    data_size = len(training_data)  

    max_arity = int(args['max_arity'])
    n_hop =  int(args['n_hop'])
    use_hypergraph  = True if args['use_hypergraph'] == 'true' else False 

    ptr= 0
    data_size = len(training_data)  
    ripple_set = data.ripple_set

    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else: 
            batch_end = data_size  

        users = [training_data[idx][0] for idx in range(ptr, batch_end)]
        items = [training_data[idx][1] for idx in range(ptr, batch_end)]

        item_list= list(data.item.keys())
        
        u_idx, i_idx, j_idx = [], [], []
        memories_h, memories_r, memories_t = [], [], [] 
        
        # for i, user in enumerate(users):
        #     neg_item = random.choice(full_items)
        #     while neg_item in items:
        #         neg_item = random.choice(full_items)
        #     neg_items.append(neg_item)
        # neg_items = np.array(neg_items)

        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            for m in range(n_negs):
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(data.item[neg_item])

        for i in range(n_hop):
            ts = []
            ts = [ripple_set[int(u)][i][2] for u in users ]
            ts = np.array(ts, dtype=int)

            memories_h.append(torch.LongTensor([ripple_set[u][i][0] for u in users]  ))
            memories_r.append(torch.LongTensor([ripple_set[u][i][1] for u in users ] ))
            
            if not use_hypergraph:
                memories_t.append(torch.LongTensor(ts)) 
            else:
                memories_t.append([])
                for j in range(max_arity):
                    memories_t[i].append(torch.LongTensor(ts[:,:,j]))

        ptr = batch_end

        u_idx  = torch.LongTensor(u_idx).cuda()
        i_idx  = torch.LongTensor(i_idx).cuda()
        j_idx  = torch.LongTensor(j_idx).cuda()

        memories_h = list(map(lambda x: x.cuda(), memories_h))
        memories_r = list(map(lambda x: x.cuda(), memories_r))

        if not use_hypergraph:
            memories_t = list(map(lambda x: x.cuda(), memories_t))
        else:
            for i in range(n_hop):
                memories_t[i] = list(map(lambda x: x.cuda(), memories_t[i]))
        yield u_idx, i_idx, j_idx, memories_h, memories_r, memories_t

def next_batch_pairwise(data,batch_size,n_negs=1):
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

        u_idx  = torch.LongTensor(u_idx).cuda()
        i_idx  = torch.LongTensor(i_idx).cuda()
        j_idx  = torch.LongTensor(j_idx).cuda()
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
