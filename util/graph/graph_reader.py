import os
import numpy as np 
import copy 
import json
from tqdm import tqdm 
import pandas as pd 
from collections import defaultdict 

def load_graph_from_file( base_dir, data_type='lastfm'):
    '''
        Xử lý file .kg 
        Đọc tìm những id đc link thì map với id của item
        Những id của entity nào ko đc link thì map với entity id mới
    '''
    print(f"Reading and Re-indexing kg file...")
    # read link and map with item-mapping 
    link_file = base_dir + f'/{data_type}/' + f'{data_type}.link'  
    kg_file = base_dir + f'/{data_type}/' + f'{data_type}.kg'  
    out_dir = base_dir +  f'/{data_type}/processed/'
    out_file =  out_dir + f'{data_type}.kg' 

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    entities = {}
    # Load file 
    data_link = np.genfromtxt(link_file, dtype='str', delimiter='\t', skip_header=1)
    data_kg = np.genfromtxt(kg_file, dtype='str', delimiter='\t')

    items = data_link[:, 0]
    item_entities = data_link[:, 1]

    # data_kg = np.genfromtxt(kg_file, dtype='str', delimiter='\t')
    relation_id = data_kg[:, 1]

    # generate item-entity mapping
    map_item_entity = {item_entities[i]: int(items[i]) for i in range(len(items))}
    # generate relation mapping
    relation_id_unique = list(set(relation_id))
    map_rel = {rel: i for i, rel in enumerate(relation_id_unique) }

    kg_data_new = []

    # generate entity mapping for row containing item 
    for _, data in enumerate(data_kg):
        head = data[0]
        tail = data[2]
        rel = data[1]

        rel_id =  map_rel[rel]

        if head in item_entities:
            head_id = map_item_entity[head] 
            if tail not in entities:
                entities[tail] = len(entities) 
                tail_id = entities[tail]
            else:       
                tail_id = entities[tail]
            kg_data_new.append([head_id, rel_id, tail_id])
                         
        if tail in item_entities:
            tail_id = map_item_entity[tail]
            if head not in entities:
                entities[head] = len(entities)
                head_id = entities[head]
            else:
                head_id = entities[head]
            kg_data_new.append([tail_id, rel_id, head_id])

    # save output file 
    processed_kg_data = np.array(kg_data_new)
    np.savetxt(out_file, processed_kg_data, fmt='%d', delimiter='\t')

    # save mapping file 
    with open(base_dir +  f'/{data_type}/processed/' + 'relation-mapping.json', 'w+') as f:
        json.dump(map_rel, f)
    with open(base_dir +  f'/{data_type}/processed/' + 'item_entity-mapping.json', 'w+') as f:
        json.dump(map_item_entity, f)
    with open(base_dir +  f'/{data_type}/processed/' + 'entity-mapping.json', 'w+') as f:
        json.dump(entities, f)

if __name__=="__main__":
    base_dir = '../../dataset/'
    data_type = 'ml-1m'
    load_graph_from_file(base_dir, data_type)

    # with open(link_file, "r") as file:
    #     # Skip the first line
    #     next(file)
    #     # Initialize an empty list to store the data
    #     data_link = []
        
    #     # Loop through each line in the file
    #     for line in file:
    #         # Split the line into columns based on the tab delimiter
    #         columns = [it.strip() for it in line.split("\t")]
    #         # Append the columns to the data list
    #         data_link.append([it for it in columns[:2]])
    #     data_link = np.array(data_link)

    # with open(kg_file, "r") as file:
    #     # Initialize an empty list to store the data
    #     data_kg = [] 
    #     # Loop through each line in the file
    #     for line in file:
    #         # Split the line into columns based on the tab delimiter
    #         columns = [it.strip() for it in line.split("\t")]
    #         # Append the columns to the data list
    #         data_kg.append([it for it in columns[:3]])
    #     data_kg = np.array(data_kg)

    # with open(inter_link, "r") as file:
    #     # Skip the first line
    #     next(file)
    #     # Initialize an empty list to store the data
    #     data = []
        
    #     # Loop through each line in the file
    #     for line in file:
    #         # Split the line into columns based on the tab delimiter
    #         columns = line.strip().split("\t")
            
    #         # Append the columns to the data list
    #         data.append([int(it) for it in columns[:3]])

    # with open(kg_file, "r") as file:
    #     # Skip the first line
    #     next(file)
    #     # Initialize an empty list to store the data
    #     data = [] 
    #     # Loop through each line in the file
    #     for line in file:
    #         # Split the line into columns based on the tab delimiter
    #         columns = line.strip().split("\t")
    #         # Append the columns to the data list
    #         data.append(columns[:3])
            
    #     np_data = np.array(data)
    #     relation_id = np_data[:,1].tolist()
    #     relation_id_unique = list(set(relation_id))
        
    #     new_rel_id = []

    #     new_item_id = []
    #     new_entity_id = []

    #     i=1
    #     map_rel = {rel: i for i, rel in enumerate(relation_id_unique) }
    #     for idx, data in enumerate(np_data):
    #         head = data[0]
    #         tail = data[2]
    #         rel = data[1]
    #         if head in map_entity.keys():
    #             if min_item <= map_entity[head] <= max_item:
    #                 # new_head_id.append(map_entity[head])
    #                 # new_tail_id.append(max_id + i)
    #                 new_item_id.append(map_entity[head])  
    #                 if tail in map_entity.keys():
    #                     new_entity_id.append(map_entity[tail])
    #                 else:
    #                     new_entity_id.append(i)
    #                     map_entity.update({tail: i})
    #                     i+= 1
    #                 new_rel_id.append(map_rel[rel])

    #         elif tail in map_entity.keys():
    #             if min_item <= map_entity[tail] <= max_item:
    #                 # new_tail_id.append(map_entity[tail])             
    #                 # new_head_id.append(max_id + i)
                    
    #                 new_item_id.append(map_entity[tail])  
    #                 if head in map_entity.keys():
    #                     new_entity_id.append(map_entity[head])
    #                 else:
    #                     new_entity_id.append(max_id+i)
    #                     map_entity.update({head: max_id + i})
    #                     i += 1

    #                 new_rel_id.append(map_rel[rel])
        
    #     # new_data_kg = np.array([new_head_id, new_tail_id, new_rel_id]).transpose()
    #     new_data_kg = np.array([new_item_id, new_entity_id, new_rel_id]).transpose()

    # with open(base_dir + '/processed/' + data_type +'.kg', 'w+') as f:
    #     for line in new_data_kg:
    #         f.write(f"{line[0]}\t{line[1]}\t{line[2]}\n")

    # with open(base_dir + '/processed/' + 'relation-entity.json', 'w+') as f:
    #     json.dump(map_rel, f)

    # with open(base_dir + '/processed/' + 'mapping-entity.json', 'w+') as f:
    #     json.dump(map_entity, f)



# Read the data from the file and store it in a 2D list
# if mode == 'inter':
#     print(f"Reading and Re-indexing inter file...")
#     process_dir = base_dir + f'/{data_type}/processed/'
#     inter_link = base_dir + f'/{data_type}/' + f'{data_type}' +'.inter'
#     if not os.path.exists(process_dir):
#         os.makedirs(process_dir)

#     with open(inter_link, "r") as file:
#         # Skip the first line
#         next(file)
#         # Initialize an empty list to store the data
#         data = []
        
#         # Loop through each line in the file
#         for line in file:
#             # Split the line into columns based on the tab delimiter
#             columns = line.strip().split("\t")
            
#             # Append the columns to the data list
#             data.append([int(it) for it in columns[:3]])

#         np_data = np.array(data)
#         user = np_data[:,0]
#         item = np_data[:,1]
#         weight = np_data[:,2]
        
#         old_item,new_item = copy.deepcopy(item),copy.deepcopy(item)
        
#         map_user = {int(user[i]): int(user[i]) for i in range(len(user))}
#         map_item = {int(old_item[i]): int(new_item[i]) for i in range(len(item))}
#         new_data_inter = np.array([user, new_item, weight]).transpose()

#         max_item = max(new_item)
#         min_item = min(new_item)
        
#     with open(process_dir + data_type +'.inter', 'w+') as f:
#         for line in new_data_inter:
#             f.write(f"{line[0]}\t{line[1]}\t{line[2]}\n")
#     with open(process_dir + 'mapping-user.json', 'w+') as f:
#         json.dump(map_user, f)
#     with open(process_dir + 'mapping-item.json', 'w+') as f:
#         json.dump(map_item, f)                

# if mode == 'kg':       
