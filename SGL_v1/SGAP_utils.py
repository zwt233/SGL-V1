from SGAP import PASCA
def getModel(model_name="SGC", 
             numfeatureP=3,
             numT=1,
             features=None, labels=None, 
             idx_train=None, idx_val=None, idx_test=None, 
             ppr=None, rw=None):
    if model_name == "SGC":
        print("\n[STEP 3]: Initialization Model Framework")
        print("| 1. Model name: {}".format(model_name))
        print("| 2. Feature Propagation times: {}".format(numfeatureP))
        print("| 3. Feature Transformation times (MLP): {}".format(numT))
        return PASCA(
                    numfeatureP=numfeatureP,    # Feature Propagation times
                    numT=numT,                  # Transformation times
                    labelProp="ppr",            # LabelProp methods???
                    labelAlpha=0,               # Label Personalized PageRank (Alpha=Alpha), Random Walk(Alpha=0)
                    featureProp="ppr",          # FeatureProp methods???
                    featureAlpha=0,             # Feature Personalized PageRank (Alpha=Alpha), Random Walk(Alpha=0)
                    message_type="last",        # Meassage aggregation -> "mean", "max", "concat", "layer weighted", "node weighted", "last"
                    features=features,          # Features
                    labels=labels,              # Labels
                    numLabelP=0,                # Label Propagation times
                    idx_train=idx_train,        # train idx
                    idx_val=idx_val,            # val idx
                    idx_test=idx_test,          # test idx
                    ppr=ppr,                    # PPR value
                    rw=rw)                      # RW value
    elif model_name == "GAMLP":
        print("\n[STEP 3]: Initialization Model Framework")
        print("| 1. Model name: {}".format(model_name))
        print("| 2. Feature Propagation times: {}".format(numfeatureP))
        print("| 3. Feature Transformation times (MLP): {}".format(numT))
        return PASCA(
                    numfeatureP=numfeatureP,    # Feature Propagation times
                    numT=numT,                  # Transformation times
                    labelProp="ppr",            # LabelProp methods???
                    labelAlpha=0,               # Label Personalized PageRank (Alpha=Alpha), Random Walk(Alpha=0)
                    featureProp="ppr",          # FeatureProp methods???
                    featureAlpha=0,             # Feature Personalized PageRank (Alpha=Alpha), Random Walk(Alpha=0)
                    message_type="node weighted",        # Meassage aggregation -> "mean", "max", "concat", "layer weighted", "node weighted", "last"
                    features=features,          # Features
                    labels=labels,              # Labels
                    numLabelP=0,                # Label Propagation times
                    idx_train=idx_train,        # train idx
                    idx_val=idx_val,            # val idx
                    idx_test=idx_test,          # test idx
                    ppr=ppr,                    # PPR value
                    rw=rw)                      # RW value

