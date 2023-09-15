import torch

class TorchGraphInterface(object):
    def __init__(self):
        pass

    @staticmethod
    def convert_sparse_mat_to_tensor(X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)
    
    @staticmethod
    def sparse_identity(n):
        indices = torch.arange(n)
        values = torch.ones(n)
        i = torch.stack((indices, indices))
        return torch.sparse_coo_tensor(i, values, size=(n, n))