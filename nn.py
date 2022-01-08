from decouple import GraphOp, FeedForwardNet, TransOp, LabelOp
import torch

class _FeatureProp(torch.nn.Module):
    def __init__(self, numfeatureP, featureProp, featureAlpha, message_type, ppr, rw):
        super(_FeatureProp, self).__init__()
        self.numfeatureP = numfeatureP
        self.featureProp = featureProp
        self.featureAlpha = featureAlpha
        self.message_type = message_type
        self.ppr = ppr
        self.rw = rw
        self.graphop = GraphOp(self.numfeatureP, self.featureProp, self.featureAlpha, self.message_type, self.ppr, self.rw)
    
    def forward(self, features):
        features_pross = self.graphop(features)
        return features_pross
    
    
class _MLPTrain(torch.nn.Module):
    def __init__(self, device, in_feats, hidden, out_feats, n_layers, dropout):
        super(_MLPTrain, self).__init__()
        self.device = device
        self.model = FeedForwardNet(
            in_feats = in_feats,
            hidden = hidden,
            out_feats = out_feats,
            n_layers = n_layers,
            dropout = dropout
        ).to(self.device)
    
    def forward(self, process_features):
        process_features = process_features.to(self.device)
        output = self.model(process_features)
        return output
    
    
class _LabelProp(torch.nn.Module):
    def __init__(self, device, labelProp, labelAlpha, numLabelP, ppr, rw):
        super(_LabelProp, self).__init__()
        self.device = device
        self.labelProp = labelProp
        self.labelAlpha = labelAlpha
        self.numLabelP = numLabelP
        self.ppr = ppr
        self.rw = rw
        self.labelop = LabelOp(self.labelProp, self.labelAlpha, self.numLabelP, self.ppr, self.rw)
    def forward(self, features):
        output_final = self.labelop(features)
        return output_final

            
    
    