from torch import nn as nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, n_head, in_dim, hi_dim, out_dim, dropout):
        """
        :param n_head: number of heads
        :param in_dim: input dimension
        :param hi_dim: hidden dimension
        :param out_dim: output dimension
        :param dropout: dropout ratio
        """
        super(GAT, self).__init__()

        self.gats = nn.ModuleList()
        self.gats.append(GATConv(in_dim, hi_dim, heads=n_head, dropout=dropout))
        self.gats.append(GATConv(hi_dim * n_head, out_dim, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout

    def reset_parameters(self):
        for gat in self.gats:
            gat.reset_parameters()

    def forward(self, x, edge_index):
        x = F.relu(self.gats[0](x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.gats[1](x, edge_index)
