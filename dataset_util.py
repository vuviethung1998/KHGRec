import random 
import os 
from data.loader import FileIO
from random import shuffle
import pandas as pd 
import numpy as np

def _create_test_file(dir, infile):
    with open(dir + infile, 'r') as f:
        lines = f.readlines()
    lines = lines[1:]    
    shuffle(lines)
    len_data = len(lines)
    
    lst_idx = [i for i in range(len_data)]
    len_train = int(len_data * 0.75)
    
    pass 

def _create_train_test_file(dir, infile):
    with open(dir + infile, 'r') as f:
        lines = f.readlines()
    lines = lines[1:]    
    shuffle(lines)
    len_data = len(lines)
    
    lst_idx = [i for i in range(len_data)]
    len_train = int(len_data * 0.75)
    id_train = random.sample(lst_idx, len_train)
    id_train = sorted(id_train)
    id_test = list(set(lst_idx).difference(set(id_train)))
    
    data_train = [ lines[id] for id in id_train]
    data_test = [ lines[id] for id in id_test]
    
    FileIO.write_file(dir, 'train.txt', data_train)
    FileIO.write_file(dir, 'test.txt', data_test)

def _create_train_test_file_missing(dir, infile, pct_missing):
    with open(dir + 'train.txt', 'r') as f:
        lines_train = f.readlines()
    lines_train = lines_train[1:]    

    with open(dir + 'test.txt', 'r') as f:
        lines_test = f.readlines()
    lines_test = lines_test[1:]    

    len_total = len(lines_train + lines_test)
    len_train = len(lines_train)
    
    len_missing = int(len_total * pct_missing) 

    data_train = lines_train[: len_train - len_missing]
    
    if not os.path.exists(dir + "missing/"):
        os.makedirs(dir + "missing/")
    
    FileIO.write_file(dir, f"missing/train_{str(int(pct_missing * 100))}.txt", data_train)
    FileIO.write_file(dir, f'missing/test_{str(int(pct_missing * 100))}.txt', lines_test)

def _create_train_test_file_coldstart(dir, infile):
    with open(dir + infile, 'r') as f:
        lines = f.readlines()
    lines = lines[1:]    
    shuffle(lines)
    len_data = len(lines)
    
    lst_idx = [i for i in range(len_data)]
    len_train = int(len_data * 0.75) 
    id_train = random.sample(lst_idx, len_train)
    id_train = sorted(id_train)
    id_test = list(set(lst_idx).difference(set(id_train)))
    data_train = [ lines[id] for id in id_train]
    data_test = [ lines[id] for id in id_test]
    
    dat_train = [ lines[id].replace('\n', '').split('\t') for id in id_train] 
    dat_test =  [ lines[id].replace('\n', '').split('\t')  for id in id_test] 
    
    df_train = pd.DataFrame(data=dat_train, columns=['user_id', 'item_id', 'rating', 'preferences'])
    df_test = pd.DataFrame(data=dat_test, columns=['user_id', 'item_id', 'rating', 'preferences'])
    
    df_train_count = df_train[['user_id', 'item_id']].groupby(['user_id'], as_index=False).count().sort_values(by='item_id')
    quantile_1 =df_train_count.item_id.quantile(0.25)
    quantile_2 =df_train_count.item_id.quantile(0.5)
    quantile_3 =df_train_count.item_id.quantile(0.75)
    
    df_group_1_users = df_train_count[df_train_count['item_id'] <= quantile_1]['user_id'].tolist()
    df_group_2_users = df_train_count[(df_train_count['item_id'] >= quantile_1) & ( df_train_count['item_id'] <= quantile_2)]['user_id'].tolist() 
    df_group_3_users = df_train_count[(df_train_count['item_id'] >= quantile_2) & ( df_train_count['item_id'] <= quantile_3)]['user_id'].tolist() 
    df_group_4_users = df_train_count[ df_train_count['item_id'] >=  quantile_3]['user_id'].tolist()
    
    df_test_group_1 = df_test[df_test['user_id'].isin(df_group_1_users)]
    df_test_group_2 = df_test[df_test['user_id'].isin(df_group_2_users)]
    df_test_group_3 = df_test[df_test['user_id'].isin(df_group_3_users)]
    df_test_group_4 = df_test[df_test['user_id'].isin(df_group_4_users)]
    
    data_tests = [df_test_group_1, df_test_group_2, df_test_group_3, df_test_group_4]
    res = []
    for df_ in data_tests:
        data_test = []
        for i, line in  df_.iterrows():
            data_test.append(f"{line['user_id']}\t{line['item_id']}\t{line['rating']}\t{line['preferences']}\n")
        res.append(data_test)
        
    if not os.path.exists(dir + "cold-start/"):
        os.makedirs(dir + "cold-start/")
    FileIO.write_file(dir, f'cold-start/train.txt', data_train)
    for idx, dat_test in enumerate(res):
        FileIO.write_file(dir, f"cold-start/test_group_{idx+1}.txt", dat_test)

def _create_kg_data(dir, infile):
    n_entities, n_relations, kg_data = FileIO.load_kg_data(dir+infile)
    return n_entities, n_relations, kg_data

def create_noisy_dict(dir):
    # read files 
    train_dir = dir + 'train.txt'
    test_dir = dir + 'test.txt'

    with open(train_dir, 'r') as f:
        lines = f.readlines()
    lines_train = lines[1:]    

    with open( test_dir, 'r') as f:
        lines = f.readlines()
    lines_test = lines[1:]   

    lines_train = [ lines_train[id].replace('\n', '').split('\t') for id, line in enumerate(lines_train)] 
    lines_test =  [ lines_test[id].replace('\n', '').split('\t')  for id, line in enumerate(lines_test)] 
    

    lines_total = lines_train + lines_test

    data = np.array(lines_total)
    items = set(data[:,1].tolist())
    
    # generate user item collection
    user_dict = {}
    for line in lines_total:
        user = line[0]
        item = line[1]
        if user not in user_dict.keys():
            user_dict.update({user: []})
        user_dict[user].append(item)

    # generate user negative item dict 
    user_neg_dict = {}
    for user in user_dict.keys():
        if user not in user_neg_dict.keys():
            user_neg_dict.update({user: []})
        user_neg_dict[user] = list(items - set(user_dict[user]))
    return user_neg_dict

def generate_noisy_data(dir, user_neg_dict, pct):
    # read files 
    train_dir = dir + 'train.txt'
    test_dir = dir + 'test.txt'

    with open(train_dir, 'r') as f:
        lines = f.readlines()
    lines_train = lines[1:]    

    with open(test_dir, 'r') as f:
        lines = f.readlines()
    lines_test = lines[1:]   

    # lines_total = lines_train + lines_test
    lines_train = [ lines_train[id].replace('\n', '').split('\t') for id, _ in enumerate(lines_train)] 
    lines_test =  [ lines_test[id].replace('\n', '').split('\t')  for id, _ in enumerate(lines_test)] 
    
    len_total = len(lines_train + lines_test)
    len_train = len(lines_train)

    # data = np.array(lines_total)
    # select keep part 
    idx_noise = random.sample(range(len_train), int(len_total * (pct)))
    idx_keep = list(set(range(len_train)) - set(idx_noise))

    data_train_keep = np.array(lines_train)[idx_keep, :]
    data_train_noise = np.array(lines_train)[idx_noise, :]

    df_noise = pd.DataFrame(data_train_noise, columns=['user', 'item', 'score', 'category'])
    def select_noise(row):
        item_noise = random.choice(user_neg_dict[row['user']])
        return item_noise
    df_noise['item'] = df_noise.apply(select_noise, axis=1)
    data_train_noise_final = df_noise[['user', 'item', 'score', 'category']].to_numpy()

    data_final = np.concatenate((data_train_keep, data_train_noise_final))
    data_final[:,:3] = data_final[:,:3].astype(int)
    shuffle(data_final)

    data_test = np.array(lines_test )
    data_test[:,:3] = data_test[:,:3].astype(int)

    if not os.path.exists(dir + "add_noise/"):
        os.makedirs(dir + "add_noise/")
    np.savetxt(dir + f"add_noise/train_{str(int(pct * 100))}.txt", data_final, fmt='%s', delimiter=',')
    np.savetxt(dir + f"add_noise/test_{str(int(pct * 100))}.txt", data_test, fmt='%s', delimiter=',')
    
if __name__ == '__main__':
    # generate noisy data
    dir = './dataset/ml-1m/'
    # user_neg_dict = create_noisy_dict(dir)
    # generate_noisy_data(dir, user_neg_dict, pct=0.1)
    # generate_noisy_data(dir, user_neg_dict, pct=0.2)
    # generate_noisy_data(dir, user_neg_dict, pct=0.3)
    # generate_noisy_data(dir, user_neg_dict, pct=0.4)
    # generate_noisy_data(dir, user_neg_dict, pct=0.5)
    
    original_file = 'ml-1m.inter'
    # _create_interaction_dataset(dir, original_file, out_file)
    # _create_train_test_file(dir, original_file)
    missing_pct = [0.1, 0.2, 0.3, 0.4, 0.5]
    for pct in missing_pct: 
        _create_train_test_file_missing(dir, original_file, pct)
    
    # _create_train_test_file_coldstart(dir, original_file)
    # _create_kg_data(dir, out_kg_file)
    