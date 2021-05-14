import numpy as np
import scipy.sparse as sp
import torch

import networkx as nx

from de.normalize import fetch_normalization, row_normalize
from de.sampler import Sampler


def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def csr_to_edgelist(matrix):
    g = nx.from_scipy_sparse_matrix(matrix.tocsr(), create_using=nx.Graph)  # dropedge sampler works as directed graph
    return torch.tensor([e for e in g.edges()]).T


def get_sampler(data, tadj):
    features = sp.csr_matrix(data.x).tolil()
    adj, features = preprocess_citation(tadj, features, 'NoNorm')
    features = np.array(features.todense())
    data.x = features
    data.adj = adj

    return Sampler(data.adj, data.x), data
