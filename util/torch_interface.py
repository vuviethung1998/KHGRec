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
    
def sparse_pow(tensor, exponent):
    """
    Raise the values of a sparse tensor to the specified power element-wise.
    :param tensor: a sparse tensor
    :param exponent: the exponent to raise the values to
    :return: a new sparse tensor with the same indices as the input tensor but with the values raised to the specified power
    """
    values = torch.pow(tensor._values(), exponent)
    return torch.sparse.FloatTensor(tensor._indices(), values, tensor.shape)

def sparse_diag(tensor):
    """
    Create a diagonal matrix from the given sparse tensor.
    :param tensor: a sparse tensor with the diagonal elements
    :return: a new sparse tensor representing a diagonal matrix
    """
    assert tensor.ndim == 1, "Input tensor should be a 1-dimensional sparse tensor."
    
    n = tensor.shape[0]
    indices = torch.arange(n, dtype=torch.long).unsqueeze(0).repeat(2, 1).to(tensor.device)
    return torch.sparse.FloatTensor(indices, tensor._values(), (n, n))

def cat_sparse_tensors(sparse_tensor1, sparse_tensor2, dim):
    indices1 = sparse_tensor1._indices()
    values1 = sparse_tensor1._values()

    indices2 = sparse_tensor2._indices()
    values2 = sparse_tensor2._values()

    # Offset indices of the second tensor along the concatenation dimension
    indices2[dim] += sparse_tensor1.shape[dim]

    # Concatenate indices and values along the concatenation dimension
    new_indices = torch.cat([indices1, indices2], dim=1)
    new_values = torch.cat([values1, values2])

    # Calculate the new shape
    new_shape = list(sparse_tensor1.shape)
    new_shape[dim] += sparse_tensor2.shape[dim]
    new_shape = tuple(new_shape)

    # Create a new sparse tensor with concatenated indices and values
    concatenated_sparse_tensor = torch.sparse.FloatTensor(new_indices, new_values, new_shape)
    return concatenated_sparse_tensor

def sparse_identity(n):
    indices = torch.arange(n)
    values = torch.ones(n)
    i = torch.stack((indices, indices))
    return torch.sparse_coo_tensor(i, values, size=(n, n))