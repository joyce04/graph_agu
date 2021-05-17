import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

'''
converted torch-geometric implementation of GCNLayer from original codes 
'''
class VGAE(nn.Module):
    """ GAE/VGAE as edge prediction model """

    def __init__(self, dim_feats, dim_h, dim_z, gae=False):
        super(VGAE, self).__init__()
        self.gae = gae
        self.gcn_base = GCNConv(dim_feats, dim_h, bias=False)  # GCNLayer(dim_feats, dim_h, 1, None, 0, bias=False)
        self.gcn_mean = GCNConv(dim_h, dim_z, bias=False)  # GCNLayer(dim_h, dim_z, 1, activation, 0, bias=False)
        self.gcn_logstd = GCNConv(dim_h, dim_z, bias=False)  # GCNLayer(dim_h, dim_z, 1, activation, 0, bias=False)

    def forward(self, adj, features):
        # GCN encoder
        hidden = F.relu(self.gcn_base(features, adj))
        self.mean = F.relu(self.gcn_mean(hidden, adj))
        if self.gae:
            # GAE (no sampling at bottleneck)
            Z = self.mean
        else:
            # VGAE
            self.logstd = F.relu(self.gcn_logstd(hidden, adj))
            gaussian_noise = torch.randn_like(self.mean)
            sampled_Z = gaussian_noise * torch.exp(self.logstd) + self.mean
            Z = sampled_Z
        # inner product decoder
        adj_logits = Z @ Z.T
        return adj_logits


def eval_edge_pred(adj_pred, val_edges, edge_labels):
    logits = adj_pred[val_edges.T]
    logits = np.nan_to_num(logits)
    roc_auc = roc_auc_score(edge_labels, logits)
    ap_score = average_precision_score(edge_labels, logits)
    return roc_auc, ap_score


# @staticmethod
# def get_lr_schedule_by_sigmoid(n_epochs, lr, warmup):
#     """ schedule the learning rate with the sigmoid function.
#     The learning rate will start with near zero and end with near lr """
#     factors = torch.FloatTensor(np.arange(n_epochs))
#     factors = ((factors / factors[-1]) * (warmup * 2)) - warmup
#     factors = torch.sigmoid(factors)
#     # range the factors to [0, 1]
#     factors = (factors - factors[0]) / (factors[-1] - factors[0])
#     lr_schedule = factors * lr
#     return lr_schedule

def pretrain_ep_net(model, optimizer, adj, features, adj_orig, norm_w, pos_weight, n_epochs, gae, val_edges, edge_labels):
    """ pretrain the edge prediction network """
    model.train(,

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        adj_logits = model(adj, features)
        loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_orig, pos_weight=pos_weight)
        if not gae:
            mu = model.mean
            lgstd = model.logstd
            kl_divergence = 0.5 / adj_logits.size(0) * (1 + 2 * lgstd - mu ** 2 - torch.exp(2 * lgstd)).sum(1).mean()
            loss -= kl_divergence
        loss.backward()
        optimizer.step()
        adj_pred = torch.sigmoid(adj_logits.detach()).cpu()
        ep_auc, ep_ap = eval_edge_pred(adj_pred, val_edges, edge_labels)
        print('EPNet pretrain, Epoch [{:3}/{}]: loss {:.4f}, auc {:.4f}, ap {:.4f}'.format(epoch + 1, n_epochs, loss.item(), ep_auc, ep_ap))


# def sample_adj(adj_logits, temperature):
#     """ sample an adj from the predicted edge probabilities of ep_net """
#     edge_probs = adj_logits / torch.max(adj_logits)
#     # sampling
#     adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=temperature, probs=edge_probs).rsample()
#     # making adj_sampled symmetric
#     adj_sampled = adj_sampled.triu(1)
#     adj_sampled = adj_sampled + adj_sampled.T
#     return adj_sampled


def normalize_adj(gnnlayer_type, adj, device):
    if gnnlayer_type == 'gcn':
        # adj = adj + torch.diag(torch.ones(adj.size(0))).to(self.device)
        adj.fill_diagonal_(1)
        # normalize adj with A = D^{-1/2} @ A @ D^{-1/2}
        D_norm = np.diag(np.power(adj.sum(1), -0.5))  # D_norm = torch.diag(torch.pow(adj.sum(1), -0.5)).to(device)
        adj = np.multiply(np.multiply(D_norm, adj), D_norm)  # adj = D_norm @ adj @ D_norm
    elif gnnlayer_type == 'gat':
        # adj = adj + torch.diag(torch.ones(adj.size(0))).to(self.device)
        adj.fill_diagonal_(1)
    elif gnnlayer_type == 'gsage':
        # adj = adj + torch.diag(torch.ones(adj.size(0))).to(self.device)
        adj.fill_diagonal_(1)
        adj = F.normalize(adj, p=1, dim=1)
    return adj
