import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def create_plot(MODEL_ID,DATASET_NAME,N_LAYERS,INTERACTION_ORDER):
    filename = MODEL_ID+"_"+DATASET_NAME+"_"+str(N_LAYERS)+"_"+str(INTERACTION_ORDER)+"_TOPK_RATIOS"
    data  = pd.read_csv(SAVE_PATH+"/"+filename+".csv")
    data = data.drop("Unnamed: 0",axis=1)
    data = data.transpose()
    means = data.mean()
    stds = data.std()

    plt.figure()
    plt.plot(means.index,means)
    plt.fill_between(means.index,means - stds,means + stds,alpha=0.2)
    plt.ylim(0,1)
    plt.title("Order: "+str(INTERACTION_ORDER))
    plt.savefig(SAVE_PATH+"/"+filename+".png")
    plt.show()


if __name__=="__main__":
    SAVE_PATH = os.path.join("..", "results", "runtime_analysis")

    INDEX = "k-SII"
    MODEL_ID = "GCN"  # one of GCN GIN GAT
    N_LAYERS = 2
    MODEL_TYPE = "GCN"
    DATASET_NAME = "Mutagenicity"

    create_plot(MODEL_ID,DATASET_NAME,N_LAYERS,INTERACTION_ORDER=1)
    create_plot(MODEL_ID,DATASET_NAME,N_LAYERS,INTERACTION_ORDER=2)
    create_plot(MODEL_ID,DATASET_NAME,N_LAYERS,INTERACTION_ORDER=3)
    create_plot(MODEL_ID,DATASET_NAME,N_LAYERS,INTERACTION_ORDER=71)