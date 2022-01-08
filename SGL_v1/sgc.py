from graphop import aug_normalized_adjacency, aug_random_walk
from modelop import *
from model_utils import load_data, sparse_mx_to_torch_sparse_tensor, logging
from SGAP import PASCA
from SGAP_utils import getModel

if __name__ == "__main__":
    # load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data(path="../data", dataset="cora")

    # FeatureProp -> Personalized PageRank
    ppr = aug_normalized_adjacency(adj)
    ppr = sparse_mx_to_torch_sparse_tensor(ppr).float()

    # FeatureProp -> Random Walk
    rw = aug_random_walk(adj)
    rw = sparse_mx_to_torch_sparse_tensor(rw).float()
    print("\n[STEP 2]: Pre-processing Graph Aggregator (Personalized PageRank and Random Walk).")
    print("| 1. Done Personalized PageRank (PPR)")
    print("| 2. Done Random Walk (RW)")
    
    getModel(model_name="SGC", 
            numfeatureP=3, numT=1,
            features=features, labels=labels,
            idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,
            ppr=ppr, rw=rw)
