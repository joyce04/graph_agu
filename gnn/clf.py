from torch import nn as nn

from gnn.fbgcn import FBGCN
from gnn.gat import GAT
from gnn.gcn import GCN
from gnn.sage import SAGE


class NodeClassifier(nn.Module):
    def __init__(self, gnn_model, loss_func):
        super(NodeClassifier, self).__init__()
        self.gnn_model = gnn_model
        self.loss_fcn = loss_func
        self.reset_parameters()

    def initialize(self):
        self.gnn_model.train()

    def evaluate(self):
        self.gnn_model.eval()

    def reset_parameters(self):
        self.gnn_model.reset_parameters()

    def forward(self, x, adj, lap, d_inv):
        return self.gnn_model(x, adj, lap, d_inv)

    def loss(self, scores, labels):
        return self.loss_fcn(scores, labels)


def generate_node_clf(gnn_type, num_feats, num_nd_classes, dropout, device):
    if gnn_type == 'gcn':
        gnn = GCN(2, num_feats, 128, num_nd_classes, dropout).to(device)
    elif gnn_type == 'graphsage':
        gnn = SAGE(2, num_feats, 128, num_nd_classes, dropout).to(device)
    elif gnn_type == 'gat':
        gnn = GAT(8, num_feats, 8, num_nd_classes, dropout).to(device)
    elif gnn_type == 'fbgcn':
        gnn = FBGCN(2, num_feats, 128, num_nd_classes, dropout).to(device)#
    return NodeClassifier(gnn, nn.CrossEntropyLoss().to(device))
