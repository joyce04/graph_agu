from torch import mm, nn, rand, tensor
from torch_geometric.nn import GATConv
import torch.nn.functional as F

class FBGAT_Layer(nn.Module):
    def __init__(self,n_head, in_dim, out_dim, dropout, concat):
        super().__init__()
      
        self.high = nn.Linear(in_dim, out_dim * n_head, bias = False)
        self.gat = GATConv(in_dim, out_dim, heads = n_head, dropout = dropout, concat = concat)
        self.dropout = dropout

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.high.weight, gain)
        
        # self.aL = nn.Parameter(rand(1))
        # self.aH = nn.Parameter(rand(1))
        self.aL = nn.Parameter(tensor(0.862))
        self.aH = nn.Parameter(tensor(0.288))

        self.gat.reset_parameters()

    def forward(self, x, edge_index, lap, d_inv):
        
        Lhp = mm(mm(d_inv, lap), d_inv)
        Hh = mm(Lhp, F.relu(self.high(x)))
        
        Hl = self.gat(x, edge_index)

        return (self.aL * Hl + self.aH * Hh)

class FBGAT(nn.Module):
    def __init__(self, n_head, in_dim, hi_dim, out_dim, dropout):
        """
        :param in_dim: input dimension
        :param hi_dim: hidden dimension
        :param out_dim: output dimension
        :param dropout: dropout ratio
        """
        super().__init__()
        self.fbgats = nn.ModuleList()
        # first layer
        self.fbgats.append(FBGAT_Layer(n_head, in_dim, hi_dim, dropout = dropout, concat = True))
        # last layer
        self.fbgats.append(FBGAT_Layer(1, hi_dim * n_head, out_dim, dropout = dropout, concat = False))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for fbgat in self.fbgats:
            fbgat.reset_parameters()

    def forward(self, x, edge_index, lap, d_inv):
        # first layer
        x = F.elu(self.fbgats[0](x, edge_index, lap, d_inv))
        x = F.dropout(x, p=self.dropout, training=self.training)
        # last layer
        return F.log_softmax(self.fbgats[-1](x, edge_index, lap, d_inv), dim=1)