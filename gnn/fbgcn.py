from torch import mm, nn, rand
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class FBGCN_Layer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
      
        self.high = nn.Linear(in_dim, out_dim, bias = False)
        self.conv = GCNConv(in_dim, out_dim)
        
    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.high.weight, gain)
        
        self.aL = nn.Parameter(rand(1))
        self.aH = nn.Parameter(rand(1))
        
        self.conv.reset_parameters()

    def forward(self, x, edge_index, lap, d_inv):
        
        Lhp = mm(mm(d_inv, lap), d_inv)
        Hh = mm(Lhp, F.relu(self.high(x)))
        
        Hl = self.conv(x, edge_index)

        return (self.aL * Hl + self.aH * Hh)

class FBGCN(nn.Module):
    def __init__(self, n_layer, in_dim, hi_dim, out_dim, dropout):
        """
        :param n_layer: number of layers
        :param in_dim: input dimension
        :param hi_dim: hidden dimension
        :param out_dim: output dimension
        :param dropout: dropout ratio
        """
        super().__init__()
        assert(n_layer > 0)
        self.num_layers = n_layer
        self.fbgcns = nn.ModuleList()
        # first layer
        self.fbgcns.append(FBGCN_Layer(in_dim, hi_dim))
        # inner layers
        for _ in range(n_layer - 2):
            self.fbgcns.append(FBGCN_Layer(hi_dim, hi_dim))
        # last layer
        self.fbgcns.append(FBGCN_Layer(hi_dim, out_dim))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for fbgcn in self.fbgcns:
            fbgcn.reset_parameters()
    def forward(self, x, edge_index, lap, d_inv):
        # first layer
        x = F.relu(self.fbgcns[0](x, edge_index, lap, d_inv))
        x = F.dropout(x, p=self.dropout, training=self.training)
        # inner layers
        if self.num_layers > 2:
            for layer in range(self.num_layers - 1):
                 x = F.relu(self.fbgcns[0](x, edge_index, lap, d_inv))
                 x = F.dropout(x, p=self.dropout, training=self.training)
        # last layer
        return F.log_softmax(self.fbgcns[-1](x, edge_index, lap, d_inv))

