import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

from util import *


def getData():
    x = pd.read_csv(getDataPath("data.csv"))
    colsToUse = [
        "Shares (millions)",
        "Offer Price",
        "Market Cap",
        "Revenues",
        "Net Income",
        "Est. $ Volume",
    ]
    x = x[colsToUse]
    return x

def addPCA(x):
    # Perform PCA to reduce the data to 2 dimensions for visualization 
    pca = PCA(n_components=2) 
    principal_components = pca.fit_transform(x) 
    principal_df = pd.DataFrame(data=principal_components, columns=['pc1', 'pc2'])
    x['pc1'] = principal_df['pc1']
    x['pc2'] = principal_df['pc2']
    return x, pca

def removeOutliers(X):
    # Remove outliers
    threshPC1 = X['pc1'].mean() + (2 * X['pc1'].std())
    threshPC2 = X['pc2'].mean() + (2 * X['pc2'].std())
    X = X[X['pc1'] < threshPC1]
    X = X[X['pc2'] < threshPC2]
    return X
