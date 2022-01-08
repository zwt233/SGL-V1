import torch
import torch.nn.functional as F
import torch.nn as nn

class Graph(nn.Module):
    def __init__(self, adj):
        super(Graph, self).__init__()
        self.adj = adj
    
    def forward(self, x):
        x = torch.spmm(self.adj, x)
        return x

class MLP(nn.Module):
    def __init__(self, nfeat, nclass, last=False):
        super(MLP, self).__init__()
        self.lr1 = nn.Linear(nfeat, nclass)
        self.last = last

    def forward(self, x):
        x = self.lr1(x)
        if not self.last:
            x = F.relu(x)
        return x    