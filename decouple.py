import torch
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn as nn
from operation import Graph, MLP

## Xunkai Li
class GraphOp(nn.Module):    
    def __init__(self, numfeatureP, featureProp, featureAlpha, message_type, ppr, rw):
        ## TODO: mean, max, concat, layer weighted, node weighted##
        # default message_type: last
        super(GraphOp, self).__init__()
        self._numP = numfeatureP      
        self._graphop = featureProp   
        self._alpha = featureAlpha     
        self._message = message_type    

        self._graph = ppr              
        if self._graphop == "rw":
            self._graph = rw
        self.graph_ops = nn.ModuleList()
        for i in range(self._numP):
            op = Graph(self._graph)
            self.graph_ops.append(op)
    
    def forward(self, s0):
        res = s0
        reserve = s0
        graph_list = [] # cache featureP result
        if self._graphop == "rw":
            for i in range(self._numP):
                res = self.graph_ops[i](res)
                graph_list.append(res)
        else:
            for i in range(self._numP):
                explore = self.graph_ops[i](res)
                res = self._alpha * reserve + (1 - self._alpha) * explore
                graph_list.append(res)
        if self._message == "last":
            res = graph_list[-1]
            
        return res
    
## Ziqi Yin
# class FeedForwardNet(nn.Module):
#     def __init__(self, in_feats, hidden, out_feats, n_layers, dropout):
#         super(FeedForwardNet, self).__init__()
#         self.layers = nn.ModuleList()
#         self.bns = nn.ModuleList()
#         self.n_layers = n_layers
#         if n_layers == 1:
#             self.layers.append(nn.Linear(in_feats, out_feats))
#         else:
#             self.layers.append(nn.Linear(in_feats, hidden))
#             self.bns.append(nn.BatchNorm1d(hidden))
#             for i in range(n_layers - 2):
#                 self.layers.append(nn.Linear(hidden, hidden))
#                 self.bns.append(nn.BatchNorm1d(hidden))
#             self.layers.append(nn.Linear(hidden, out_feats))
#         if self.n_layers > 1:
#             self.prelu = nn.PReLU()
#             self.dropout = nn.Dropout(dropout)
#         #self.norm=bns
#         self.reset_parameters()
#     def reset_parameters(self):
#         gain = nn.init.calculate_gain("relu")
#         for layer in self.layers:
#             nn.init.xavier_uniform_(layer.weight, gain=gain)
#             nn.init.zeros_(layer.bias)
#     def forward(self, x):
#         for layer_id, layer in enumerate(self.layers):
#             x = layer(x)
#             if layer_id < self.n_layers -1:
#                 #if self.norm:
#                 x = self.dropout(self.prelu(self.bns[layer_id](x)))
#                 #else:
#                 #    x = self.dropout(self.prelu(x))
#         return x

## Xunkai Li
class FeedForwardNet(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout):
        ## TODO: mean, max, concat, combine with TransOp##
        # default message_type: last
        super(FeedForwardNet, self).__init__()
        self._numT = n_layers               
        self._drop = dropout            
        # trans
        self.trans_ops = nn.ModuleList()
        for i in range(self._numT):
            if i == 0:
                if i == self._numT - 1:
                    op = MLP(in_feats, hidden, True)
                else:
                    op = MLP(in_feats, hidden)
            elif i == self._numT - 1:
                op = MLP(hidden, out_feats, True)
            else:
                op = MLP(hidden, hidden)
            self.trans_ops.append(op)
        
    def forward(self, res):
        # trans
        for i in range(self._numT):
            if i == 0:
                res = F.dropout(res, self._drop, training=self.training)
                res = self.trans_ops[i](res)
                residual_1 = res
                residual_2 = res
            else:
                if i % 2 == 1:
                    #residual_1 = res
                    res = F.dropout(res, self._drop, training=self.training)
                    res = self.trans_ops[i](res + residual_2)
                else:
                    #residual_2 = res
                    res = F.dropout(res, self._drop, training=self.training)
                    res = self.trans_ops[i](res + residual_1)
            
        logits = F.softmax(res, dim=1)
        return logits

## TransOp ??? ##
## TODO: weighted ##
class TransOp(nn.Module):
    def __init__(self, numfeatureP, numT, message_type, feat_dim, hid_dim, num_classes, dropout):
        super(TransOp, self).__init__()
        self._numP = numfeatureP        # feature Propogate times 
        self._message = message_type    # message aggregation methods
        self._numT = numT               # feature Transform times
        self._drop = dropout            # dropout ratio
        self.dropout = nn.Dropout(dropout)
        # trans
        self.trans_ops = nn.ModuleList()
        for i in range(self._numT):
            op = FeedForwardNet(feat_dim, hid_dim, num_classes, self._numT,dropout)
            self.trans_ops.append(op)
        #weight
        ## TODO ##
        # if self._message == "node weighted":
        # elif self._message == "layer weighted":
        #     if numT==0:
        #         self.lr_att1=nn.Linear(feat_dim, hid_dim)
        #     else:
        #         self.lr_att1=nn.Linear(hid_dim, hid_dim)                         
    def forward(self, res, features):
        for i in range(len(res)):
             res[i] = self.trans_ops[i](res[i])                 
        if self._message == "node weighted":
            right_1 = sum([torch.mul(F.sigmoid(self.gate[i]), res[i]) for i in range(self._numP)])
            res = right_1
            
        if self._message == "node weighted":
            num_node = features.shape[0]
            res = torch.split(res, features.shape[1], dim = 1)
            attention_scores = [torch.sigmoid(self.lr_att1(x).view(num_node,1)) for x in res]
            W = torch.cat(attention_scores, dim=1)
            W = F.softmax(W,1)
            
            right_1 = torch.mul(res[0], W[:,0].view(num_node,1)) 
            for i in range(1, self._numP):
                right_1 = right_1 + torch.mul(res[i], W[:,i].view(num_node,1)) 
            res = right_1
        return res
    

class LabelOp(nn.Module):
    def __init__(self, labelProp, labelAlpha, numLabelP, ppr, rw):
        super(LabelOp, self).__init__()
        self._numLabelP = numLabelP     # labelP times
        self._labelProp = labelProp     # labelP methods
        self._labelAlpha = labelAlpha   # labelP alpha
            
        # label_ops
        self._label = ppr
        if self._labelProp == "rw":
            self._label = rw
        self.label_ops = nn.ModuleList()
        for i in range(self._numLabelP):
            op = Graph(self._label)
            self.label_ops.append(op)
        
    def forward(self, res):
        # labels
        label_reserve = res
        if self._labelProp == "rw":
            for i in range(self._numLabelP):
                res = self.label_ops[i](res)
        else:
            for i in range(self._numLabelP):
                explore = self.label_ops[i](res)
                res = self._labelAlpha * label_reserve + (1 - self._labelAlpha) * explore    
        return res
    

