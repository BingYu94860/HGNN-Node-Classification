import numpy as np
import scipy.sparse as sp


def norm_DF(features):
    F = sp.coo_matrix(features, dtype='float32')
    def fmask(x): return np.array(x) != 0
    def np_power(x, p): return np.power(x, p, where=fmask(x))
    D = np_power(np.sum(np.abs(F), 1), -1)
    return F.multiply(D)


def norm_DAD(A):
    A = sp.coo_matrix(A, dtype='float32')
    def fmask(x): return np.array(x) != 0
    def np_power(x, p): return np.power(x, p, where=fmask(x))
    D0 = np_power(np.sum(np.abs(A), 1), -0.5)
    D1 = np_power(np.sum(np.abs(A), 0), -0.5)
    return A.multiply(D0).multiply(D1)


def norm_DADsm(adj):
    A_sm = adj + sp.eye(adj.shape[0])
    adj_norm = norm_DAD(A_sm)
    return adj_norm


def norm_DvH_WDe_HDv(H, W=None):
    H = sp.coo_matrix(H, dtype='float32')
    num_nodes, num_hyperedges = H.shape
    if W is None:
        W = np.ones([num_hyperedges])
    W = np.reshape(W, [1, num_hyperedges]).astype('float32')
    def fmask(x): return np.array(x) != 0
    def np_power(x, p): return np.power(x, p, where=fmask(x))
    De = np_power(np.sum(H, 0), -1)  # (1, num_hyperedges)
    Dv = np_power(np.sum(H.multiply(W), 1), -0.5)  # (num_nodes, 1)
    WDe = np.multiply(W, De)  # WDe = W @ De^(-1)
    DvH = H.multiply(Dv)  # DvH = Dv^(-0.5) @ H
    return DvH.multiply(WDe).dot(DvH.T)  # DvH @ WDe @ HDv
