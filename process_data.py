import numpy as np 
import pandas as pd 
from collections import defaultdict

# Process cf data
def read_cf(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        if "\t" not in tmps:
            if ',' in tmps:
                inters = [i for i in tmps.split(",")]
            else:
                inters = [i for i in tmps.split(" ")]
        else:
            inters = [i for i in tmps.split("\t")]
        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([int(u_id), int(i_id)])

    return np.array(inter_mat)


def read_triplets(file_name):
    can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
    can_triplets_np = np.unique(can_triplets_np, axis=0)
    return can_triplets_np


def process_cf_data(in_dir, out_dir):
    train_np = read_cf(in_dir+ '/train.txt') 
    test_np = read_cf(in_dir+ '/test.txt')
    
    with open(out_dir + '/train.txt', 'w+') as f:
        for line in train_np:
            f.write(f"{line[0]}\t{line[1]}\t1\n")
    
    with open(out_dir + '/test.txt', 'w+') as f:
        for line in test_np:
            f.write(f"{line[0]}\t{line[1]}\t1\n")

# Process KG data
def process_kg_data(in_dir, out_dir, dataset):
    kg_data = read_triplets(in_dir +'/kg_final.txt')
    
    with open(out_dir + f'/{dataset}.kg', 'w+') as f:
        f.write('head_id:token\trelation_id:token\ttail_id:token\n')
        for line in kg_data:
            f.write(f"{line[0]}\t{line[1]}\t{line[2]}\n")

if  __name__ == "__main__":
    dataset= 'alibaba-fashion'
    in_dir = f'./dataset/{dataset}/raw'
    out_dir = f'./dataset/{dataset}'
    process_cf_data(in_dir,out_dir)
    process_kg_data(in_dir, out_dir, dataset)