import networkx as nx
import torch


def csr_to_edgelist(matrix):
    g = nx.from_scipy_sparse_matrix(matrix.tocsr(), create_using=nx.Graph)  # dropedge sampler works as directed graph
    return torch.tensor([e for e in g.edges()]).T