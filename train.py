import torch
from utils import load_data, accuracy
from model import SGC
import torch.optim as optim
import torch.nn.functional as F
from utils import aug_normalized_adjacency, aug_random_walk, sparse_mx_to_torch_sparse_tensor
import numpy as np
import os

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data(path="data", dataset="cora")
    labels = labels.to(device)
    
    ## Pre-process
    print("\n[STEP 2]: Pre-processing Graph Aggregator (Personalized PageRank and Random Walk).")
    print("| 1. Done Personalized PageRank (PPR)")
    print("| 2. Done Random Walk (RW)")
    # FeatureProp -> Personalized PageRank
    ppr = aug_normalized_adjacency(adj)
    ppr = sparse_mx_to_torch_sparse_tensor(ppr).float()
    # FeatureProp -> Random Walk
    rw = aug_random_walk(adj)
    rw = sparse_mx_to_torch_sparse_tensor(rw).float()
    
    ## Create model
    print("\n[STEP 3]: Initialization Model Framework")
    print("| 1. Model name: {}".format("SGC"))
    print("| 2. Feature Propagation times: {}".format(3))
    print("| 3. Feature Transformation times (MLP): {}".format(1))
    print("| 4. Feature Propagation times: {}".format(2))
    model = SGC(
        in_feats = features.shape[1], 
        hidden = 128, 
        out_feats = (labels.max()+1).item(), 
        n_layers = 1, 
        dropout = 0.8, 
        device = device,
        ppr = ppr,
        rw = rw
    )
    
    ## Decoupled model training
    print("\n[STEP 4]: Model training")
    print("| 1. Execute featureProp")
    # 1. featureProp
    featuresP_output = model.FeatureProp(features)
    featuresP_output = featuresP_output.to(device)
    test_acc = []
    best_acc = 0
    
    print("| 2. Execute MLPtrain")
    print("|   -> Train Model 10 times (200epochs)")
    for i in range(10):
        ## Initialize the model per training
        model = SGC(
            in_feats = features.shape[1], 
            hidden = 128, 
            out_feats = (labels.max()+1).item(), 
            n_layers = 1, 
            dropout = 0.8, 
            device = device,
            ppr = ppr,
            rw = rw
        )
        record = {}
        optimizer = optim.Adam(model.parameters(), lr=1e-1, weight_decay=5e-4)
        # 2. mlp model train
        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            output = model(featuresP_output)
            loss_train = F.nll_loss(output[idx_train].log(), labels[idx_train])
            acc_train = accuracy(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            
            # model eval
            model.eval()
            output = model(featuresP_output)
            acc_val = accuracy(output[idx_val], labels[idx_val])
            acc_test = accuracy(output[idx_test], labels[idx_test])
            record[acc_val.item()] = acc_test.item()
            if acc_val > best_acc:
                best_acc = acc_val
                torch.save(model, './best.pt')
        
        # process results
        bit_list = sorted(record.keys())
        bit_list.reverse()
        cur_acc = record[bit_list[0]]
        print("|     Model train times:{}, Test acc:{}".format(i, round(cur_acc, 4)))
        test_acc.append(cur_acc)
    print("|   -> Validation Model: Mean Test acc: {}, Std Test acc: {}".format(round(np.mean(test_acc), 4), round(np.std(test_acc, ddof=1), 4)))
    
    # 3. labelProp
    print("| 3. Execute labelProp")
    if os.path.isfile("./best.pt"):
        model = torch.load('./best.pt')
        model.eval()
        output = model(featuresP_output).cpu()
        final_val_acc = accuracy(output[idx_val], labels[idx_val]).item()
        final_test_acc = accuracy(output[idx_test], labels[idx_test]).item()
        print(f'|   -> Original Val acc: {final_val_acc:.4f}, Original Test acc: {final_test_acc:.4f}')
        output_final = model.LabelProp(output)
        acc_test = accuracy(output_final[idx_test], labels[idx_test]).item()
        print(f'|   -> Final Test acc: {acc_test:.4f}')
    else:
        raise Exception("no local model")


    