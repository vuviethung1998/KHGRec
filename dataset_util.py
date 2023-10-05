import random 
import os 
from data.loader import FileIO
from random import shuffle
import pandas as pd 


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
    with open(dir + infile, 'r') as f:
        lines = f.readlines()
    lines = lines[1:]    
    shuffle(lines)
    len_data = len(lines)
    
    lst_idx = [i for i in range(len_data)]
    len_train = int(len_data * 0.75 * pct_missing) 
    id_train = random.sample(lst_idx, len_train)
    id_train = sorted(id_train)
    id_test = list(set(lst_idx).difference(set(id_train)))
    data_train = [ lines[id] for id in id_train]
    data_test = [ lines[id] for id in id_test]
    if not os.path.exists(dir + "missing/"):
        os.makedirs(dir + "missing/")
    
    FileIO.write_file(dir, f"missing/train_{str(int(pct_missing * 100))}.txt", data_train)
    FileIO.write_file(dir, f'missing/test_{str(int(pct_missing * 100))}.txt', data_test)

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
    import pdb; pdb.set_trace()
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

if __name__ == '__main__':
    dir = './dataset/ml-1m/'
    original_file = 'ml-1m.inter'
    # _create_interaction_dataset(dir, original_file, out_file)
    # _create_train_test_file(dir, original_file)
    # missing_pct = [0.2, 0.4, 0.6, 0.8]
    # for pct in missing_pct: 
    #     _create_train_test_file_missing(dir, original_file, pct)
    
    _create_train_test_file_coldstart(dir, original_file)
    # _create_kg_data(dir, out_kg_file)
    