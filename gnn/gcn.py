from torch import nn as nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv

"""
    Based on Kipf, Thomas N., and Max Welling. "Semi-supervised classification with graph convolutional networks." arXiv preprint arXiv:1609.02907 (2016).
"""


class GCN(nn.Module):
    def __init__(self, n_layer, in_dim, hi_dim, out_dim, dropout):
        """
        :param n_layer: number of layers
        :param in_dim: input dimension
        :param hi_dim: hidden dimension
        :param out_dim: output dimension
        :param dropout: dropout ratio
        """
        super(GCN, self).__init__()
        assert (n_layer > 0)

        self.num_layers = n_layer
        self.gcns = nn.ModuleList()
        # first layer
        self.gcns.append(GCNConv(in_dim, hi_dim))
        # inner layers
        for _ in range(n_layer - 2):
            self.gcns.append(GCNConv(hi_dim, hi_dim))
        # last layer
        self.gcns.append(GCNConv(hi_dim, out_dim))
        self.dropout = dropout

    def reset_parameters(self):
        for gcn in self.gcns:
            gcn.reset_parameters()

    def forward(self, x, edge_index):
        # first layer
        x = F.relu(self.gcns[0](x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        # inner layers
        if self.num_layers > 2:
            for layer in range(1, self.num_layers - 1):
                x = F.relu(self.gcns[layer](x, edge_index))
                x = F.dropout(x, p=self.dropout, training=self.training)

        # last layer
        return self.gcns[self.num_layers - 1](x, edge_index)
