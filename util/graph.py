import networkx as nx
import scipy
import torch
from torch import zeros, from_numpy
from tqdm import tqdm


def csr_to_edgelist(matrix):
    g = nx.from_scipy_sparse_matrix(matrix.tocsr(), create_using=nx.Graph)  # dropedge sampler works as directed graph
    return torch.tensor([e for e in g.edges()]).T


def lap_dinv(edge_index, num_nodes):
    adj = zeros((num_nodes, num_nodes))
    degree = zeros((num_nodes, num_nodes))
    for i in tqdm(range(edge_index.shape[1])):
        first = edge_index[0][i]
        second = edge_index[1][i]
        adj[first][second] = 1
        adj[second][first] = 1
    for i in tqdm(range(num_nodes)):
        degree[i][i] = sum(adj[i][:])
    lap = degree - adj
    inter = scipy.linalg.fractional_matrix_power(degree, (-1 / 2))
    d_inv = from_numpy(inter)
    return (lap, d_inv)
