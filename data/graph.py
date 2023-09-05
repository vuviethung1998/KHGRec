import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse import diags

class Graph(object):
    def __init__(self):
        pass

    @staticmethod
    def normalize_graph_mat(adj_mat):
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1))
        if shape[0] == shape[1]:
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
        else:
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
        return norm_adj_mat

    @staticmethod
    def torch_normalize_graph_mat(adj_mat):
        shape = adj_mat.shape
        rowsum = torch.sum(adj_mat, dim=1)
        if shape[0] == shape[1]:
            d_inv = torch.pow(rowsum, -0.5).flatten()
            d_inv[torch.isinf(d_inv)] = 0.
            d_mat_inv = diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
        else:
            d_inv = torch.pow(rowsum, -1).flatten()
            d_inv[torch.isinf(d_inv)] = 0.
            d_mat_inv = diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
        return norm_adj_mat

    def convert_to_laplacian_mat(self, adj_mat):
        pass
    