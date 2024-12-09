import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

from analysis.util import *


def getData(includeClusters = False):
    x = pd.read_csv(getDataPath("data.csv"))
    colsToUse = [
        "Shares (millions)",
        "Offer Price",
        "Market Cap",
        "Revenues",
        "Net Income",
    ]
    if includeClusters:
        colsToUse.extend([
            "Industry_Cluster",
            "Month_Cluster"
        ])
    s = x["Symbol"]
    x = x[colsToUse]
    return x[colsToUse], s

def addPCA(x):
    # Perform PCA to reduce the data to 2 dimensions for visualization 
    pca = PCA(n_components=2) 
    principal_components = pca.fit_transform(x) 
    principal_df = pd.DataFrame(data=principal_components, columns=['pc1', 'pc2'])
    x['pc1'] = principal_df['pc1']
    x['pc2'] = principal_df['pc2']
    return x, pca

def removeOutliers(x):
    # Remove outliers
    threshPC1 = x['pc1'].mean() + (0.1 * x['pc1'].std())
    threshPC2 = x['pc2'].mean() + (0.1 * x['pc2'].std())
    x = x[x['pc1'] < threshPC1]
    x = x[x['pc2'] < threshPC2]
    return x


if __name__ == "__main__":
    data = pd.read_csv(getDataPath("data.csv"))
    stocks = pd.read_csv(getDataPath("stocks.csv"), header=[0,1], index_col=[0])

    
    print(data, len(data))