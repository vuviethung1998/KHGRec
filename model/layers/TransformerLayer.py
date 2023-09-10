import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np

"""
    Graph Transformer Layer
    
"""

"""
    Util functions
"""
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}
    return func

def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}

    return func


"""
    Single Attention Head
"""

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        
        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        
    
    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score')) #, edges)
        g.apply_edges(scaled_exp('score', np.sqrt(self.out_dim)))

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))
    
    def forward(self, g, h):
        
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        
        # Reshaping into [num_nodes, num_heads, feat_dim] to 
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
        
        self.propagate_attention(g)
        
        head_out = g.ndata['wV']/g.ndata['z']
        
        return head_out
    