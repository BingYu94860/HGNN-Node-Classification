import numpy as np
import scipy.sparse as sp
from scipy.sparse.coo import isspmatrix_coo
import tensorflow as tf

def to_sparse_tensor(X):
    """ 轉換成稀疏張量, dtype='folat32'

    Args:
        X ([Sparse Matrix / np.ndarray]): [description]

    Returns:
        [Sparse Tensor]: [description]
    """
    coo = sp.coo_matrix(X) 
    indices = np.vstack((coo.row, coo.col)).transpose()
    Y = tf.SparseTensor(indices, np.float32(coo.data), coo.shape)
    Y = tf.sparse.reorder(Y)
    return Y


def sparse_to_tuple(sparse_mx):
    # tf.SparseTensor(*sparse_to_tuple(x))
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx
