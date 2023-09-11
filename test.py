import torch

def sparse_identity(n):
    indices = torch.arange(n)
    values = torch.ones(n)
    i = torch.stack((indices, indices))
    return torch.sparse_coo_tensor(i, values, size=(n, n))

n = 5
I = sparse_identity(n)
print(I)
