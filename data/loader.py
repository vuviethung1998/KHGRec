import os.path
from os import remove
from re import split
import numpy as np 
import pandas as pd 

class FileIO(object):
    def __init__(self):
        pass

    @staticmethod
    def write_file(dir, file, content, op='w'):
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(dir + file, op) as f:
            f.writelines(content)

    @staticmethod
    def delete_file(file_path):
        if os.path.exists(file_path):
            remove(file_path)

    @staticmethod
    def load_data_set(file, rec_type='graph'):
        data = []
        with open(file) as f:
            next(f)
            for line in f:
                if '\t' not in line:
                    items = split(',', line.strip())
                else:
                    items = split('\t', line.strip())
                user_id = int(items[0])
                item_id = int(items[1])

                weight = 1
                data.append([user_id, item_id, float(weight)])
        return data

    @staticmethod
    def load_user_list(file):
        user_list = []
        print('loading user List...')
        with open(file) as f:
            for line in f:
                user_list.append(line.strip().split()[0])
        return user_list

    @staticmethod
    def load_social_data(file):
        social_data = []
        print('loading social data...')
        with open(file) as f:
            for line in f:
                items = split('\t', line.strip())
                user1 = items[0]
                user2 = items[1]
                if len(items) < 3:
                    weight = 1
                else:
                    weight = float(items[2])
                social_data.append([user1, user2, weight])
        return social_data
    
    @staticmethod 
    def load_kg_data(filename):
        # load Kg from file 
        kg_df = pd.read_csv(filename, sep='\t', header=None, engine='python', skiprows=1, \
                              names= ['head_id:token','relation_id:token','tail_id:token'])
        kg_df = kg_df.rename(columns={
            'head_id:token': 'h',
            'relation_id:token': 'r',
            'tail_id:token': 't'
        })
        # kg_np = kg_df.to_numpy()
        # n_entity = len(set(kg_np[:, 0]) & set(kg_np[:, 2]))
        # n_relation = len(set(kg_np[:, 1]))
        return kg_df

# def load_cf(filename):
#     # load user and item from file 
#     user = []
#     item = []
#     user_dict = dict()

#     lines = open(filename, 'r').readlines()

#     data = [] 
#     for l in lines:
#         tmp = l.strip()
#         inter = [int(i) for i in tmp.split()]

#         if len(inter) > 1:
#             user_id, item_ids = inter[0], inter[1:]
#             item_ids = list(set(item_ids))

#             for item_id in item_ids:
#                 user.append(user_id)
#                 item.append(item_id)
#                 data.append([user_id, item_id, 1.0])

#             user_dict[user_id] = item_ids
#     user = np.array(user, dtype=np.int32)
#     item = np.array(item, dtype=np.int32)
#     # data = data[:100000]
#     return user, item, user_dict, data 

# def construct_kg(file):
#     datas = [] 
#     with open(file) as f:
#         for line in f:
#             items = split('\t', line.strip())
#             head_id = int(items[0])
#             relation_id = int(items[1])
#             tail_id = int(items[2])
#             datas.append([head_id, relation_id, tail_id])
#         kg_np = np.array(datas)
#         kg_df = pd.DataFrame(datas, columns=['head_id', 'relation_id', 'tail_id'])
#     return kg_df, kg_np

if __name__=="__main__":
    kg = FileIO.load_kg_data("./dataset/lastfm/lastfm.kg")
    import pdb; pdb.set_trace()