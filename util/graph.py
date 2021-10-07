import networkx as nx
import torch
from torch_geometric.utils import to_networkx


def csr_to_edgelist(matrix):
    g = nx.from_scipy_sparse_matrix(matrix.tocsr(), create_using=nx.Graph)  # dropedge sampler works as directed graph
    return torch.tensor([e for e in g.edges()]).T


def homophily_measure(data):
    # graph = to_networkx(data, to_undirected=True, node_attrs=['y'])
    # homophily = 0
    # for node in graph.nodes:
    #     smth = 0
    #     self_loop = False
    #     for neigh in graph.neighbors(node):
    #         if neigh == node:
    #             self_loop = True
    #             continue
    #         if graph.nodes[node]['y'] == graph.nodes[neigh]['y']:
    #             smth += 1
    #     num_neigh = graph.degree(node)
    #     if self_loop == True:
    #         self_loop = False
    #         num_neigh = num_neigh - 1
    #     if num_neigh == 0:
    #         smth_rate = 0
    #     else:
    #         smth_rate = smth / num_neigh
    #     homophily = homophily + smth_rate
    # homophily = homophily / data.num_nodes
    return 1., 1.