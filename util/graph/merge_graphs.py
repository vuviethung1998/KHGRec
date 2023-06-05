import numpy as np
import os
import argparse
import networkx as nx
from networkx.readwrite import json_graph
import json
import pdb

def parse_args():
    parser = argparse.ArgumentParser(description="Merge two graphs into one.")
    parser.add_argument("--base_dir", default="./data/", help='Path to save data')
    parser.add_argument("--type", default="movie", choices=['book', 'movie'], help='Dataset prefix')
    parser.add_argument('--prefix1', default="interaction")
    parser.add_argument('--prefix2', default="kg")
    parser.add_argument('--out_prefix', default="merge")
    return parser.parse_args()

def load_graph(prefix):
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    res = json_graph.node_link_data(G)

    id_map = json.load(open(prefix + "-id_map.json"))
    return G, id_map, res

def generate_graphs(base_dir, datatype, prefix1, prefix2):
    prefix1 = base_dir + datatype + '/graphsage/' + prefix1
    prefix2 = base_dir + datatype + '/graphsage/' + prefix2 
    G1, id_map1, res1 = load_graph(prefix1)
    G2, id_map2, res2 = load_graph(prefix2)

    new_nodes_ids1 = np.arange(len(G1.nodes()))
    new_nodes_ids2 = np.arange(len(G1.nodes()), len(G1.nodes())+len(G2.nodes()))
    new_nodes_ids1 = list(map(str, new_nodes_ids1)) # source nodes ids
    new_nodes_ids2 = list(map(str, new_nodes_ids2)) # target nodes ids

    new_id_map = {}
    new_id_map_g1 = {}
    new_id_map_g2 = {}
    for node_id in new_nodes_ids1:
        new_id_map[node_id] = int(node_id)
        new_id_map_g1[node_id] = int(node_id)
    for node_id in new_nodes_ids2:
        new_id_map[node_id] = int(node_id)
        new_id_map_g2[node_id]  = int(node_id)

    # merge-G
    new_nodes = []
    new_nodes_g1 = []
    new_nodes_g2 = []
    for idx, node in enumerate(res1["nodes"]):
        node["id"] = new_nodes_ids1[idx]
        new_nodes.append(node)
        new_nodes_g1.append(node)
    for idx, node in enumerate(res2["nodes"]):
        node["id"] = new_nodes_ids2[idx]
        new_nodes.append(node)
        new_nodes_g2.append(node)
    new_links = []
    new_links_g1 = []
    new_links_g2 = []

    for link in res1["links"]:
        new_links.append({
            'source': str(new_id_map[new_nodes_ids1[int(link["source"])]]),
            'target': str(new_id_map[new_nodes_ids1[int(link["target"])]])
        })
        new_links_g1.append({
            'source': str(new_id_map[new_nodes_ids1[int(link["source"])]]),
            'target': str(new_id_map[new_nodes_ids1[int(link["target"])]])
        })

    for link in res2["links"]:
        new_links.append({
            'source': str(new_id_map[new_nodes_ids2[int(link["source"])]]),
            'target': str(new_id_map[new_nodes_ids2[int(link["target"])]])
        })
        new_links_g2.append({
            'source': str(new_id_map[new_nodes_ids2[int(link["source"])]]),
            'target': str(new_id_map[new_nodes_ids2[int(link["target"])]])
        })
    new_res = res1
    new_res["nodes"] = new_nodes
    new_res["links"] = new_links

    new_res_g1 = res1 
    new_res_g1['nodes'] = new_nodes_g1
    new_res_g1['links'] = new_links_g1

    new_res_g2 = res2
    new_res_g2['nodes'] = new_nodes_g2
    new_res_g2['links'] = new_links_g2
    return new_res, new_id_map, new_res_g1, new_id_map_g1, new_res_g2, new_id_map_g2, new_nodes_ids1, new_nodes_ids2

def save_graph(args, res, id_map, res_g1, id_map_g1, res_g2, id_map_g2, source_ids, target_ids):
    base_dir, out_prefix, prefix1, prefix2= args.base_dir, args.out_prefix, args.prefix1, args.prefix2
    out_dir = base_dir + args.type

    if not os.path.exists(out_dir+"/graphsage"):
        os.makedirs(out_dir+"/graphsage/")
    if not os.path.exists(out_dir+"/edgelist"):
        os.makedirs(out_dir+"/edgelist")
    if not os.path.exists(out_dir+"/dictionaries"):
        os.makedirs(out_dir+"/dictionaries")
    with open(out_dir +"/graphsage/" + out_prefix +"-G.json", "w") as file:
        file.write(json.dumps(res))
    with open(out_dir+"/graphsage/" + out_prefix+"-id_map.json", "w") as file:
        file.write(json.dumps(id_map))

    with open(out_dir +"/graphsage/" + prefix1 + "-remap" +"-G.json", "w") as file:
        file.write(json.dumps(res_g1))
    with open(out_dir+"/graphsage/" +  prefix1 + "-remap" + "-id_map.json", "w") as file:
        file.write(json.dumps(id_map_g1))

    with open(out_dir +"/graphsage/" + prefix2 + "-remap" +"-G.json", "w") as file:
        file.write(json.dumps(res_g2))
    with open(out_dir+"/graphsage/" +  prefix2 + "-remap" + "-id_map.json", "w") as file:
        file.write(json.dumps(id_map_g2))

    np.save(out_dir+"/graphsage/"+out_prefix+"-source_ids.npy", source_ids)
    np.save(out_dir+"/graphsage/"+out_prefix+"-target_ids.npy", target_ids)

    nx.write_edgelist(json_graph.node_link_graph(res),
        path=out_dir + "/edgelist/" + out_prefix + ".edgelist" , delimiter=" ", data=['weight'])

    # with open(out_dir + "/dictionaries/" + "gt_like.gt", 'w') as f:
    #     for src, trg in new_gt_like.items():
    #         f.write("{} {}\n".format(src, trg))
    #     f.close()

    # with open(out_dir + "/dictionaries/" + "gt_dislike.gt", 'w') as f:
    #     for src, trg in new_gt_dislike.items():
    #         f.write("{} {}\n".format(src, trg))
    #     f.close()

    print("Graph has been saved to {}".format(out_dir))

def test_graph(args):
    prefix = args.base_dir + args.type + '/graphsage/' + args.out_prefix
    G, _,_ = load_graph(prefix)
    nodes = G.nodes()
    nodes = sorted(list(map(int, nodes)))
    nodes_expected = np.arange(len(nodes)).tolist()
    print("Test graph id map:", nodes == nodes_expected)

if __name__ == "__main__":
    args = parse_args()
    print(args)
    res, id_map, res_g1, id_map_g1, res_g2, id_map_g2, source_ids, target_ids = generate_graphs(args.base_dir, args.type, args.prefix1, args.prefix2)
    save_graph(args, res, id_map, res_g1, id_map_g1, res_g2, id_map_g2, source_ids, target_ids)
    test_graph(args)
