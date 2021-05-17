import copy
import pickle

import numpy as np
import torch
from scipy import sparse as sp

from gaug.util import scipysp_to_pytorchsp
from gaug.vae import pretrain_ep_net, VGAE
from util.graph import csr_to_edgelist


class GAug:
    def __init__(self, pretrain):
        self.gae = True
        self.pretrain = pretrain
        self.updated_edges = None
        self.ep_net = None

    def sample_graph_det(self, adj_orig, A_pred, remove_pct, add_pct):
        if remove_pct == 0 and add_pct == 0:
            return copy.deepcopy(adj_orig)
        orig_upper = sp.triu(adj_orig, 1)
        n_edges = orig_upper.nnz
        edges = np.asarray(orig_upper.nonzero()).T
        if remove_pct:
            n_remove = int(n_edges * remove_pct / 100)
            pos_probs = A_pred[edges.T[0], edges.T[1]]
            e_index_2b_remove = np.argpartition(pos_probs, n_remove)[:n_remove]
            mask = np.ones(len(edges), dtype=bool)
            mask[e_index_2b_remove] = False
            # Check removal edges
            edges_pred = edges[mask]
        else:
            edges_pred = edges

        if add_pct:
            n_add = int(n_edges * add_pct / 100)
            # deep copy to avoid modifying A_pred
            A_probs = np.array(A_pred)
            # make the probabilities of the lower half to be zero (including diagonal)
            A_probs[np.tril_indices(A_probs.shape[0])] = 0
            # make the probabilities of existing edges to be zero
            A_probs[edges.T[0], edges.T[1]] = 0
            all_probs = A_probs.reshape(-1)
            e_index_2b_add = np.argpartition(all_probs, -n_add)[-n_add:]
            new_edges = []
            for index in e_index_2b_add:
                i = int(index / A_probs.shape[0])
                j = index % A_probs.shape[0]
                new_edges.append([i, j])
            edges_pred = np.concatenate((edges_pred, new_edges), axis=0)
        adj_pred = sp.csr_matrix((np.ones(len(edges_pred)), edges_pred.T), shape=adj_orig.shape)
        adj_pred = adj_pred + adj_pred.T
        return adj_pred

    def get_pretrained_edges(self, dataset, data, gaug_param):
        if self.pretrain:
            A_pred = pickle.load(open(f'./gaug/edge_probabilities/{dataset}_graph_5_logits.pkl', 'rb'))
            adj_pred = self.sample_graph_det(data.adj, A_pred, gaug_param['removal_rate'], gaug_param['add_rate'])
            self.updated_edges = csr_to_edgelist(adj_pred).type(torch.int64)

    def train_predict_edges(self, adj, feature, labels, device, gaug_param):
        adj_matrix, adj_norm, adj_orig = self.preprocess_adj(adj)
        norm_w = adj_orig.shape[0] ** 2 / float((adj_orig.shape[0] ** 2 - adj_orig.sum()) * 2)
        pos_weight = torch.FloatTensor([float(adj_orig.shape[0] ** 2 - adj_orig.sum()) / adj_orig.sum()]).to(device)

        # edge data
        if labels.size(0) > 5000:
            edge_frac = 0.01
        else:
            edge_frac = 0.1
        adj_matrix = sp.csr_matrix(adj_matrix)
        n_edges_sample = int(edge_frac * adj_matrix.nnz / 2)

        # sample negative edges
        neg_edges = self.sample_neg_edges(adj_matrix, n_edges_sample)
        # sample positive edges
        pos_edges = self.sample_pos_edges(adj_matrix, n_edges_sample)
        # when number edges in the training set is not sufficient
        if pos_edges.shape[0] < n_edges_sample:
            n_edges_sample = pos_edges.shape[0]
            neg_edges = neg_edges[:n_edges_sample]
        val_edges = np.concatenate((pos_edges, neg_edges), axis=0)
        edge_labels = np.array([1] * n_edges_sample + [0] * n_edges_sample)

        if not self.ep_net:
            self.ep_net = VGAE(feature.shape[1], 128, 32, gae=self.gae)
            learning_rate = 1e-2
            self.optimizer = torch.optim.Adam(self.ep_net.parameters(), lr=learning_rate)
        else:  # TODO is retraining better?
            for layer in self.ep_net.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        pretrain_ep_net(self.ep_net, self.optimizer, adj_norm, feature, adj_orig, norm_w, pos_weight, gaug_param['ep'], self.gae, val_edges, edge_labels)

        # predict
        with torch.no_grad():
            self.ep_net.eval()
            adj_logits = self.ep_net(adj_norm, feature)
            # adj_new = normalize_adj('gcn', sample_adj(adj_logits, TEMP), device)

            adj_pred = self.sample_graph_det(adj, adj_logits, gaug_param['removal_rate'], gaug_param['add_rate'])
            self.updated_edges = csr_to_edgelist(adj_pred).type(torch.int64)

    def sample_pos_edges(self, adj_matrix, n_edges_sample):
        nz_upper = np.array(sp.triu(adj_matrix, k=1).nonzero()).T
        np.random.shuffle(nz_upper)
        pos_edges = nz_upper[:n_edges_sample]
        return pos_edges

    def sample_neg_edges(self, adj_matrix, n_edges_sample):
        neg_edges = []
        added_edges = set()
        while len(neg_edges) < n_edges_sample:
            i = np.random.randint(0, adj_matrix.shape[0])
            j = np.random.randint(0, adj_matrix.shape[0])
            if i == j:
                continue
            if adj_matrix[i, j] > 0:
                continue
            if (i, j) in added_edges:
                continue
            neg_edges.append([i, j])
            added_edges.add((i, j))
            added_edges.add((j, i))
        neg_edges = np.asarray(neg_edges)
        return neg_edges

    def preprocess_adj(self, adj):
        adj_matrix = adj
        if not isinstance(adj_matrix, sp.coo_matrix):
            adj_matrix = sp.coo_matrix(adj_matrix)
        adj_matrix.setdiag(1)
        adj_orig = scipysp_to_pytorchsp(adj_matrix).to_dense()
        degree_mat_inv_sqrt = sp.diags(np.power(np.array(adj_matrix.sum(1)), -0.5).flatten())
        adj_norm = degree_mat_inv_sqrt @ adj_matrix @ degree_mat_inv_sqrt
        # adj_norm = scipysp_to_pytorchsp(adj_norm) # adjust to fit pytorch-geometric
        adj_norm = csr_to_edgelist(adj_norm).type(torch.int64)
        return adj_matrix, adj_norm, adj_orig
