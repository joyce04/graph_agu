import random

import torch
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork
from torch_geometric import transforms as T
import networkx as nx
import numpy as np
from torch_geometric.utils.convert import to_networkx
import math


def train_test_split_nodes(data, train_ratio=0.2, val_ratio=0.05, test_ratio=0.1, class_balance=True):
    r"""Splits nodes into train, val, test masks
    """
    n_nodes = data.num_nodes
    train_mask, ul_train_mask, val_mask, test_mask = torch.zeros(n_nodes), torch.zeros(n_nodes), torch.zeros(n_nodes), torch.zeros(n_nodes)

    n_tr = round(n_nodes * train_ratio)
    n_val = round(n_nodes * val_ratio)
    n_test = round(n_nodes * test_ratio)

    train_samples, rest = [], []

    if class_balance:
        unique_cls = list(set(data.y.numpy()))
        n_cls = len(unique_cls)
        cls_samples = [n_tr // n_cls + (1 if x < n_tr % n_cls else 0) for x in range(n_cls)]

        for cls, n_s in zip(unique_cls, cls_samples):
            cls_ss = (data.y == cls).nonzero().T.numpy()[0]
            cls_ss = np.random.choice(cls_ss, len(cls_ss), replace=False)
            train_samples.extend(cls_ss[:n_s])
            rest.extend(cls_ss[n_s:])

        train_mask[train_samples] = 1
        assert (sorted(train_samples) == list(train_mask.nonzero().T[0].numpy()))
        rand_indx = np.random.choice(rest, len(rest), replace=False)
        # train yet unlabeled
        ul_train_mask[rand_indx[n_val + n_test:]] = 1

    else:
        rand_indx = np.random.choice(np.arange(n_nodes), n_nodes, replace=False)
        train_mask[rand_indx[n_val + n_test:n_val + n_test + n_tr]] = 1
        # train yet unlabeled
        ul_train_mask[rand_indx[n_val + n_test + n_tr:]] = 1

    val_mask[rand_indx[:n_val]] = 1
    test_mask[rand_indx[n_val:n_val + n_test]] = 1

    data.train_mask = train_mask.to(torch.bool)
    data.val_mask = val_mask.to(torch.bool)
    data.test_mask = test_mask.to(torch.bool)
    data.ul_train_mask = ul_train_mask.to(torch.bool)
    return data


def train_test_split_edges(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    r"""Modified version of torch_geometric train_test_split_edges
    Instead of train_neg_adj_mask, return randomly selected train_neg_edge_index

    Splits the edges of a :obj:`torch_geometric.data.Data` object
    into positive and negative train/val/test edges, and adds attributes of
    `train_pos_edge_index`, `train_neg_edge_index`, `val_pos_edge_index`,
    `val_neg_edge_index`, `test_pos_edge_index`, and `test_neg_edge_index`
    to :attr:`data`.

    Args:
        data (Data): The data object.
        val_ratio (float, optional): The ratio of positive validation
            edges. (default: :obj:`0.05`)
        test_ratio (float, optional): The ratio of positive test
            edges. (default: :obj:`0.1`)

    :rtype: :class:`torch_geometric.data.Data`
    """

    assert 'batch' not in data  # No batch-mode.

    num_nodes = data.num_nodes
    row, col = data.edge_index

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    n_tr = int(math.floor(train_ratio * row.size(0)))
    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]

    if train_ratio == 1.0:
        data.val_pos_edge_index = torch.stack([row, col], dim=0)
        data.test_pos_edge_index = torch.stack([row, col], dim=0)
        data.train_pos_edge_index = torch.stack([row, col], dim=0)
    else:
        r, c = row[:n_v], col[:n_v]
        data.val_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        data.test_pos_edge_index = torch.stack([r, c], dim=0)

        r, c = row[n_v + n_t:n_v + n_t + n_tr], col[n_v + n_t:n_v + n_t + n_tr]
        data.train_pos_edge_index = torch.stack([r, c], dim=0)

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = random.sample(range(neg_row.size(0)), min(n_v + n_t + n_tr, neg_row.size(0)))
    perm = torch.tensor(perm)
    perm = perm.to(torch.long)
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    neg_adj_mask[neg_row, neg_col] = 0
    data.train_neg_adj_mask = neg_adj_mask

    if train_ratio == 1.0:
        data.train_neg_edge_index = torch.stack([row, col], dim=0)
        data.val_neg_edge_index = torch.stack([row, col], dim=0)
        data.test_neg_edge_index = torch.stack([row, col], dim=0)
    else:
        row, col = neg_row[:n_tr], neg_col[:n_tr]
        data.train_neg_edge_index = torch.stack([row, col], dim=0)

        row, col = neg_row[n_tr:n_tr + n_v], neg_col[n_tr:n_tr + n_v]
        data.val_neg_edge_index = torch.stack([row, col], dim=0)

        row, col = neg_row[n_tr + n_v:n_tr + n_v + n_t], neg_col[n_tr + n_v:n_tr + n_v + n_t]
        data.test_neg_edge_index = torch.stack([row, col], dim=0)

    return data


def get_unlabeled_nodes(data):
    ul_train_mask = torch.zeros(data.num_nodes)
    selected_nodes = torch.cat((data.train_mask.nonzero().T, data.val_mask.nonzero().T, data.test_mask.nonzero().T), dim=1)[0]
    b = torch.Tensor([i for i in range(data.train_mask.shape[0])]).type(torch.int64)
    ul_train_mask[list(set(b.numpy()).difference(set(selected_nodes.numpy())))] = 1
    data.ul_train_mask = ul_train_mask
    return data


def dataset_split(file_loc='./dataset/', dataset_name='cora', split_type='public', subset_ratio=0.1, edge_split=False):
    if dataset_name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=file_loc, name=dataset_name, transform=T.NormalizeFeatures(), split=split_type)
    elif dataset_name in ['cornell', 'texas', 'wisconsin']:  # TODO
        dataset = WebKB(root=file_loc, name=dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root=file_loc, name=dataset_name, transform=T.NormalizeFeatures())
    else:
        raise Exception('dataset not available...')

    if split_type == 'public':
        data = dataset[0]
        data = get_unlabeled_nodes(data)
    elif split_type == 'full':
        data = dataset[0]
    else:
        # dataset = Planetoid(root=file_loc, name=dataset_name, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.train_mask = data.val_mask = data.test_maks = None
        data = train_test_split_nodes(data, subset_ratio, val_ratio=0.2, test_ratio=0.2, class_balance=True)
        if edge_split:
            data = train_test_split_edges(data, subset_ratio, val_ratio=0.2, test_ratio=0.2)

    # self-loop
    if edge_split:
        data.train_index = data.train_pos_edge_index
    else:
        data.train_index = data.edge_index
    data.train_index = add_self_loops(data.train_index, data.num_nodes)
    adj, deg = build_graph(data, split_type)
    data.adj = adj
    data.degree = deg
    return dataset, data


def add_self_loops(edge_index, num_nodes):
    self_loop = torch.Tensor([[el, el] for el in range(num_nodes)])
    return torch.cat([edge_index.T, self_loop]).T.type(torch.int64)


def get_graph(data, edges):
    g = nx.Graph()
    g.add_nodes_from([v for v in range(data.train_mask.shape[0])])
    g.add_edges_from([(el[0], el[1]) for el in edges])
    return g


def get_adj(g):
    adj = nx.adjacency_matrix(g)
    return adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)


def build_graph(data, data_split):
    if data_split in ['public', 'full']:
        data.edge_index = data.train_index
        g = to_networkx(data, to_undirected=True)
    else:
        g = get_graph(data, data.train_index.T.numpy())
    adj = nx.adjacency_matrix(g)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return adj, np.sum(adj, axis=1)
