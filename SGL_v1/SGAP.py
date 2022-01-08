import torch.nn as nn
from modelop import Graph, MLP
import torch
import torch.nn.functional as F
from model_utils import accuracy
import time
import torch.optim as optim
import numpy as np
class GraphOp(nn.Module):
    
    def __init__(self, numfeatureP, featureProp, featureAlpha, message_type, ppr, rw):
        super(GraphOp, self).__init__()
        self._numP = numfeatureP        # featureP times 
        self._graphop = featureProp     # featureP methods
        self._alpha = featureAlpha      # featureP alpha
        self._message = message_type    # message aggregation methods

        # graph_ops
        self._graph = ppr               # default ppr
        if self._graphop == "rw":
            self._graph = rw
        self.graph_ops = nn.ModuleList()
        for i in range(self._numP):
            op = Graph(self._graph)
            self.graph_ops.append(op)
    
    def forward(self, s0):
        # graph
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
        
        # message
        if self._message == "mean": # mean
            res = torch.mean(torch.stack(graph_list), 0)
        elif self._message == "max": # max
            res = torch.max(torch.stack(graph_list), 0)[0]
        elif self._message == "concat": # concat
            res = torch.cat(graph_list, 1)
        elif self._message == "layer weighted": # layer weighted
            res = torch.cat(graph_list, 1)
        elif self._message == "node weighted": # node weighted
            res = torch.cat(graph_list, 1)
        elif self._message == "last": # last
            res = graph_list[-1]
            
        return res


class TransOp(nn.Module):
    def __init__(self, numfeatureP, numT, message_type, feat_dim, hid_dim, num_classes, dropout):
        super(TransOp, self).__init__()
        self._numP = numfeatureP        # featureP times 
        self._message = message_type    # message aggregation methods
        self._numT = numT               # featureT times
        self._drop = dropout            # dropout ratio
        # trans
        self.trans_ops = nn.ModuleList()
        for i in range(self._numT):
            if i == 0:
                if self._message == "mean" or self._message == "max":
                    if i == self._numT - 1:
                        op = MLP(feat_dim, hid_dim, True)
                    else:
                        op = MLP(feat_dim, hid_dim)
                elif self._message == "concat":
                    if i == self._numT - 1:
                        op = MLP(feat_dim * self._numP, hid_dim, True)
                    else:
                        op = MLP(feat_dim * self._numP, hid_dim)
                elif self._message == "layer":
                    self.gate = torch.nn.Parameter(1e-5*torch.randn(self._numP), requires_grad=True)
                    if i == self._numT - 1:
                        op = MLP(feat_dim, hid_dim, True)
                    else:
                        op = MLP(feat_dim, hid_dim)
                elif self._message == "node weighted":
                    self.lr_att1 = nn.Linear(feat_dim, 1)
                    if i == self._numT - 1:
                        op = MLP(feat_dim, hid_dim, True)
                    else:
                        op = MLP(feat_dim, hid_dim)
                elif self._message == "last":
                    if i == self._numT - 1:
                        op = MLP(feat_dim, hid_dim, True)
                    else:
                        op = MLP(feat_dim, hid_dim)
            elif i == self._numT - 1:
                op = MLP(hid_dim, num_classes, True)
            else:
                op = MLP(hid_dim, hid_dim)
            self.trans_ops.append(op)
        
    def forward(self, res, features):
        # trans
        if self._message == "layer weighted":
            res = torch.split(res, features.shape[1], dim = 1)
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
    

class LabelOp(nn.Module):
    def __init__(self, labelProp, labelAlpha, numLabelP):
        super(LabelOp, self).__init__()
        self._numLabelP = numLabelP     # labelP times
        self._labelProp = labelProp     # labelP methods
        self._labelAlpha = labelAlpha   # labelP alpha

        # label_ops
        self.label_ops = nn.ModuleList()
        for i in range(self._numLabelP):
            op = Graph(self._labelProp)
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
    
    
def FeatureProp(numfeatureP, featureProp, featureAlpha, message_type, features, ppr, rw):
    graphop = GraphOp(numfeatureP, featureProp, featureAlpha, message_type, ppr, rw)
    t1 = time.time()
    features_pross = graphop(features)
    t2 = time.time()
    total_time = t2 - t1
    return features_pross, total_time
    

def _MLPTrain(epoch, model, optimizer, features_pross, features, record, idx_train, idx_val, idx_test, labels):
    model.train()
    optimizer.zero_grad()
    output = model(features_pross, features)
    loss_train = F.nll_loss(output[idx_train].log(), labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    if epoch % 20 == 0:
        print("|       loss_train:{}, acc_train :{}".format(loss_train, acc_train))

    loss_train.backward()
    optimizer.step()
    
    model.eval()
    output = model(features_pross, features)
    
    acc_val = accuracy(output[idx_val], labels[idx_val])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    if epoch % 20 == 0:
        print("|       acc_val :{}".format(acc_val))
        print("|       acc_test :{}".format(acc_test))
    
    record[acc_val.item()] = acc_test.item()
    return acc_val
    

def MLPTrain(numfeatureP, numT, message_type, features_pross, features, labels, idx_train, idx_val, idx_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_class = (labels.max()+1).item()
    features_pross = features_pross.to(device)
    test_acc = []
    best_acc = 0
    print("|   -> Train Model 10 times (200epochs)")
    for i in range(10):
        t_total = time.time()
        record = {}
        model = TransOp(numfeatureP, numT, message_type, features.shape[1], 128, n_class, 0.8).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-1, weight_decay=5e-4)
        for epoch in range(200):
            acc_val = _MLPTrain(epoch, model, optimizer, features_pross, features, record, idx_train, idx_val, idx_test, labels.to(device))
            if acc_val > best_acc:
                best_acc = acc_val
                torch.save(model, './best.pt')
                
        bit_list = sorted(record.keys())
        bit_list.reverse()
        cur_acc = record[bit_list[0]]
        print("|     Model train times:{}, Test acc:{}".format(i, round(cur_acc, 4)))
        test_acc.append(cur_acc)
    print("|   -> Validation Model: Mean Test acc: {}, Std Test acc: {}".format(round(np.mean(test_acc), 4), round(np.std(test_acc, ddof=1), 4)))
    

def LabelProp(features_pross, features, idx_val, idx_test, labels, labelProp, labelAlpha, numLabelP):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_1 = time.time()
    features_pross = features_pross.to(device)
    
    model = torch.load('./best.pt')
    model.eval()
    output = model(features_pross, features).cpu()
    t_2 = time.time()
    
    final_val_acc = accuracy(output[idx_val], labels[idx_val]).item()
    final_test_acc = accuracy(output[idx_test], labels[idx_test]).item()
    print(f'|   -> Original Val acc: {final_val_acc:.4f}, Original Test acc: {final_test_acc:.4f}')
    
    t_3 = time.time()
    labelop = LabelOp(labelProp, labelAlpha, numLabelP)
    output_final = labelop(output)
    t_4 = time.time()
    
    acc_test = accuracy(output_final[idx_test], labels[idx_test]).item()
    print(f'|   -> Final Test acc: {acc_test:.4f}')
    total_time = (t_2 - t_1) + (t_4 - t_3)
    return acc_test, total_time


def PASCA(numfeatureP, 
          numT, 
          labelProp, 
          labelAlpha, 
          featureProp,
          featureAlpha, 
          message_type, 
          numLabelP, 
          features, labels, 
          idx_train, idx_val, idx_test, 
          ppr, rw):
    print("\n[STEP 4]: Model training")
    print("| 1. Execute featureProp")
    features_pross, feature_time = FeatureProp(numfeatureP=numfeatureP, 
                                               featureProp=featureProp,
                                               featureAlpha=featureAlpha, 
                                               message_type=message_type, 
                                               features=features, 
                                               ppr=ppr, 
                                               rw=rw)
    print("| 2. Execute MLPtrain")
    print(features_pross)
    MLPTrain(numfeatureP=numfeatureP, 
             numT=numT, 
             message_type=message_type, 
             features_pross=features_pross, 
             features=features, labels=labels, 
             idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)
    
    print("| 3. Execute labelProp")
    acc_res, time_res = LabelProp(features_pross=features_pross, 
                                  features=features,
                                  labelProp=labelProp, 
                                  labelAlpha=labelAlpha, 
                                  numLabelP=numLabelP,
                                  idx_val=idx_val, idx_test=idx_test, labels=labels)
    result = dict()
    result['objs'] = np.stack([-acc_res, (feature_time + time_res - 0.058)/0.3141], axis=-1)
    return result
