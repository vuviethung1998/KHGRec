from __future__ import print_function, division
import numpy as np
import random
import json
import sys
import os
import argparse
from shutil import copyfile
import networkx as nx
from networkx.readwrite import json_graph
import pdb

def parse_args():
    parser = argparse.ArgumentParser(description="Generate subgraphs of a network.")
    parser.add_argument('--base_dir', default='./data/', help='Path to save data')
    parser.add_argument('--type', default="book", choices=['book', 'movie'], help='Dataset prefix')
    parser.add_argument('--mode', default='interaction', choices=['interaction', 'kg'], help='Mode to choose')
    parser.add_argument('--min_node', type=int, default=100, help='minimum node for subgraph to be kept')
    return parser.parse_args()

def main(args):
    if args.mode == 'interaction':
        fname = 'ratings_final'
        node_cols = [0, 1]
        attr_cols = 2
    elif args.mode == 'kg':
        fname = 'kg_final'
        node_cols = [0, 2]
        attr_cols = 1

    original_data = np.load(args.base_dir + args.type + '/original/' + fname + ".npy" ).tolist()
    
    G = nx.DiGraph()
    for i, row in enumerate(original_data):
        G.add_edge(row[node_cols[0]], row[node_cols[1]], attr_dict=row[attr_cols])
    G_data = json_graph.node_link_data(G)

    if isinstance(G_data['nodes'][0], int):
        def conversion(n): return int(n)
    else:
        def conversion(n): return n['id']
    mapping = {conversion(G_data['nodes'][i]):str(G_data['nodes'][i]['id']) for i in range(len(G_data['nodes']))}
    G = nx.relabel_nodes(G, mapping)
    # print("Original graph info: ")
    # print(nx.info(G))

    print("Saving graph...")
    save_new_graph(G, args.base_dir, args.type, args.mode)
    return

def save_new_graph(G, base_dir, data_type, prefix):
    output_dir =  base_dir + data_type 

    nodes = G.nodes
    if not os.path.exists(output_dir + "/edgelist/"):
        os.makedirs(output_dir+ "/edgelist/")
    if not os.path.exists(output_dir + "/graphsage/"):
        os.makedirs(output_dir + "/graphsage/")

    nx.write_edgelist(G, path = output_dir + "/edgelist/"  + prefix + ".edgelist" , delimiter=" ", data=False)

    output_prefix = output_dir + "/graphsage/" + prefix

    print("Saving new id map")
    new_idmap = {node: i for i, node in enumerate(nodes)}
    with open(output_prefix + '-id_map.json', 'w') as outfile:
        json.dump(new_idmap, outfile)

    print("Saving new graph")
    num_nodes = len(G.nodes)
    rand_indices = np.random.permutation(num_nodes)
    train = rand_indices[:int(num_nodes * 0.81)]
    val = rand_indices[int(num_nodes * 0.81):int(num_nodes * 0.9)]
    test = rand_indices[int(num_nodes * 0.9):]

    id_map = new_idmap
    res = json_graph.node_link_data(G)
    # print(res['links'])
    res['nodes'] = [
            {
                'id': str(node['id']),
                'val': id_map[str(node['id'])] in val,
                'test': id_map[str(node['id'])] in test
            }
            for node in res['nodes']]

    res['links'] = [
            {
                'source': link['source'],
                'target': link['target']
            }
            for link in res['links']]

    with open(output_prefix + "-G.json", 'w') as outfile:
        json.dump(res, outfile)

    print("DONE!")

if __name__ == "__main__":
    args = parse_args()
    print(args)
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    main(args)