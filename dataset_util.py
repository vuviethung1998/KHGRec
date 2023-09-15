import random 
from data.loader import FileIO
from random import shuffle

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

def _create_kg_data(dir, infile):
    n_entities, n_relations, kg_data = FileIO.load_kg_data(dir+infile)
    return n_entities, n_relations, kg_data

if __name__ == '__main__':
    dir = './dataset/alibaba-fashion/'
    original_file = 'alibaba-fashion.inter'
    # _create_interaction_dataset(dir, original_file, out_file)
    _create_train_test_file(dir, original_file)
    # _create_kg_data(dir, out_kg_file)
    