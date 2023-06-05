import os.path
from os import remove
from re import split
import numpy as np 
import pandas as pd 
import collections

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
    def load_data_set(file, rec_type):
        if rec_type == 'graph':
            data = []
            with open(file) as f:
                for line in f:
                    items = split('\t', line.strip())
                    user_id = int(items[0])
                    item_id = int(items[1])
                    weight = float(items[2])
                    data.append([user_id, item_id, float(weight)])

        if rec_type == 'sequential':
            data = {}
            with open(file) as f:
                for line in f:
                    items = split(':', line.strip())
                    seq_id = items[0]
                    data[seq_id]=items[1].split()
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
    def load_kg_data(file):
        _, kg_np = construct_kg(file)
        n_entity = len(set(kg_np[:, 2]))
        n_relation = len(set(kg_np[:, 1])) 
        return n_entity, n_relation, kg_np

def construct_kg(file):
    datas = [] 
    with open(file) as f:
        for line in f:
            items = split('\t', line.strip())
            head_id = int(items[0])
            relation_id = int(items[1])
            tail_id = int(items[2])
            datas.append([head_id, relation_id, tail_id])
        kg_np = np.array(datas)
        kg_df = pd.DataFrame(datas, columns=['head_id', 'relation_id', 'tail_id'])
    return kg_df, kg_np

if __name__=="__main__":
    n_ent, n_rel, kg = FileIO.load_kg_data("./dataset/lastfm/filtered_kg.txt")