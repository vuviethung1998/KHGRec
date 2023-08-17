import torch 
import numpy as np
import random
import scipy.sparse as sp
from math import floor

class GraphAugmentor(object):
    def __init__(self):
        pass

    @staticmethod
    def node_dropout(sp_adj, drop_rate):
        """Input: a sparse adjacency matrix and a dropout rate."""
        adj_shape = sp_adj.get_shape()
        # print(adj_shape)
        row_idx, col_idx = sp_adj.nonzero()
        drop_user_idx = random.sample(range(adj_shape[0]), int(adj_shape[0] * drop_rate))
        drop_item_idx = random.sample(range(adj_shape[1]), int(adj_shape[1] * drop_rate))
        indicator_user = np.ones(adj_shape[0], dtype=np.float32)
        indicator_item = np.ones(adj_shape[1], dtype=np.float32)
        indicator_user[drop_user_idx] = 0.
        indicator_item[drop_item_idx] = 0.
        diag_indicator_user = sp.diags(indicator_user)
        diag_indicator_item = sp.diags(indicator_item)
        mat = sp.csr_matrix(
            (np.ones_like(row_idx, dtype=np.float32), (row_idx, col_idx)),
            shape=(adj_shape[0], adj_shape[1]))
        mat_prime = diag_indicator_user.dot(mat).dot(diag_indicator_item)
        return mat_prime

    @staticmethod
    def edge_dropout(sp_adj, drop_rate):
        """Input: a sparse user-item adjacency matrix and a dropout rate."""
        adj_shape = sp_adj.get_shape()
        edge_count = sp_adj.count_nonzero()
        row_idx, col_idx = sp_adj.nonzero()
        keep_idx = random.sample(range(edge_count), int(edge_count * (1 - drop_rate)))
        user_np = np.array(row_idx)[keep_idx]
        item_np = np.array(col_idx)[keep_idx]
        edges = np.ones_like(user_np, dtype=np.float32)
        dropped_adj = sp.csr_matrix((edges, (user_np, item_np)), shape=adj_shape)
        return dropped_adj
    
    @staticmethod
    def edge_dropout_index(edge_index, drop_rate):
        aug_edge_index1 = drop_edges(edge_index, drop_rate)
        aug_edge_index2 = drop_edges(edge_index, drop_rate)
        return (aug_edge_index1, aug_edge_index2)

def drop_edges(edge_index, aug_ratio=0.4):
    # This function will randomly drop some edges of the original graph to generate a variant augmented graph.
    num_edges = len(edge_index[0])
    drop_num = int(num_edges * aug_ratio)

    idx_perm = np.random.permutation(num_edges)
    edge_idx1 = edge_index[0][idx_perm]
    edge_idx2 = edge_index[1][idx_perm]

    edges_keep1 = edge_idx1[drop_num:]
    edges_keep2 = edge_idx2[drop_num:]

    aug_edge_index = torch.stack([edges_keep1, edges_keep2], dim=0)

    return aug_edge_index

class SequenceAugmentor(object):
    def __init__(self):
        pass

    @staticmethod
    def item_crop(seq, seq_len, crop_ratio):
        augmented_seq = np.zeros_like(seq)
        augmented_pos = np.zeros_like(seq)
        aug_len = []
        for i, s in enumerate(seq):
            start = random.sample(range(seq_len[i]-floor(seq_len[i]*crop_ratio)),1)[0]
            crop_len = floor(seq_len[i]*crop_ratio)+1
            augmented_seq[i,:crop_len] =seq[i,start:start+crop_len]
            augmented_pos [i,:crop_len] = range(1,crop_len+1)
            aug_len.append(crop_len)
        return augmented_seq, augmented_pos, aug_len

    @staticmethod
    def item_reorder(seq, seq_len, reorder_ratio):
        augmented_seq = seq.copy()
        for i, s in enumerate(seq):
            start = random.sample(range(seq_len[i]-floor(seq_len[i]*reorder_ratio)),1)[0]
            np.random.shuffle(augmented_seq[i,start:floor(seq_len[i]*reorder_ratio)+start+1])
        return augmented_seq

    @staticmethod
    def item_mask(seq, seq_len, mask_ratio, mask_idx):
        augmented_seq = seq.copy()
        for i, s in enumerate(seq):
            to_be_masked = random.sample(range(seq_len[i]), floor(seq_len[i]*mask_ratio))
            augmented_seq[i, to_be_masked] = mask_idx
        return augmented_seq

