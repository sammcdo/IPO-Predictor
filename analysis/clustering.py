import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

from clustering_util import *


np.random.seed(42)

### Visualize Data
x, _ = getData()
x, _ = addPCA(x)

# plot the ipos
plt.scatter(x['pc1'], x['pc2'], cmap='viridis')
plt.title('Visualize IPOs')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()


### Remove Outliers
x, _ = getData()
x, _ = addPCA(x)
x = removeOutliers(x)
print(x)

# Add plot the ipos
plt.scatter(x['pc1'], x['pc2'], cmap='viridis')
plt.title('Visualize IPOs w/o Outliers')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()


### Find Number of Clusters
x, _ = getData()
x, _ = addPCA(x)
x = removeOutliers(x)

# Calculate WCSS for different number of clusters 
wcss = []
for i in range(1, 16):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

# show the elbow plot
plt.plot(range(1, 16), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.show()


### Create Clusters
x, s = getData(includeClusters=False)
x, pca = addPCA(x)
x["symbol"] = s
x = removeOutliers(x)
y = x[["pc1", "pc2", "symbol"]]
x.drop(columns=["pc1", "pc2", "symbol"], inplace=True)

# Create KMeans instance with 2 clusters
kmeans = KMeans(n_clusters=2)
kmeans.fit(x)

# Get the cluster labels
centroids = pca.transform(kmeans.cluster_centers_) # scale to same as components
x['Cluster'] = kmeans.labels_
for c in y.columns:
    x[c] = y[c]

old, s = getData(includeClusters=True)
old["symbol"] = s
old = old[old["symbol"].isin(x['symbol'])]
x["Industry_Cluster"] = old["Industry_Cluster"]
x["Month_Cluster"] = old["Month_Cluster"]

# Plot the centroids
plt.scatter(x['pc1'], x['pc2'], c=x['Cluster'], cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='red', marker='X')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# save output
x.to_csv(writeFilePath("clustered.csv"))
