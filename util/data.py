import torch
from torch_geometric.datasets import Planetoid
from torch_geometric import transforms as T
import networkx as nx
import numpy as np
from torch_geometric.utils.convert import to_networkx


def dataset_split(file_loc='../dataset/', dataset_name='cora', split_type='public'):
    if dataset_name in ['cora', 'citeseer', 'pubmed']:
        if split_type in ['public', 'full']:
            dataset = Planetoid(root=file_loc, name=dataset_name, transform=T.NormalizeFeatures(), split=split_type)
            data = dataset[0]
            data.edge_index = add_self_loops(data.edge_index, data.num_nodes)
            adj, deg = build_graph(data)
            data.adj = adj
            data.degree = deg
    return dataset

    raise Exception('Given dataset not available...')


def add_self_loops(edge_index, num_nodes):
    self_loop = torch.Tensor([[el, el] for el in range(num_nodes)])
    return torch.cat([edge_index.T, self_loop]).T.type(torch.int64)


def build_graph(data):
    graph = to_networkx(data, to_undirected=True)
    adj = nx.adjacency_matrix(graph)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return adj, np.sum(adj, axis=1)
