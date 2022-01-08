import torch
from nn import _FeatureProp, _MLPTrain, _LabelProp

class SGC(torch.nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout, device, ppr, rw):
        super(SGC, self).__init__()
        self.device = device
        self.in_feats = in_feats
        self.hidden = hidden
        self.out_feats = out_feats
        self.n_layers = n_layers
        self.dropout = dropout
        self.ppr = ppr
        self.rw = rw
        self.FeatureProp = _FeatureProp(
            numfeatureP = 3, 
            featureProp = "ppr", 
            featureAlpha = 0, 
            message_type = "last", 
            ppr = self.ppr, 
            rw = self.rw
        )
        self.MLPTrain = _MLPTrain(
            device = self.device, 
            in_feats = self.in_feats, 
            hidden = self.hidden, 
            out_feats = self.out_feats, 
            n_layers = self.n_layers, 
            dropout = self.dropout
        )
        self.LabelProp = _LabelProp(
            device = self.device, 
            labelProp = "ppr", 
            labelAlpha = 0, 
            numLabelP = 2,
            ppr = self.ppr,
            rw = self.rw
        )
    def forward(self, data):
        train_output = self.MLPTrain(data)
        return train_output