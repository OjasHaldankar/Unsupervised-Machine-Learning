#!/usr/bin/env python
# coding: utf-8

# ## `Unsupervised ML  :  Principle Component Analysis (PCA) + Clustering`

# Importing Libraries
import pandas as pd 
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 

# Importing Dataset
data = pd.read_csv("wine.csv")
data.head(10)

data.describe()

# Selecting the data required for analysis
data1 = data.iloc[:, 1:]
df = data1.values
df

# Standardizing the data
df_std = scale(df)
df_std

# Initializing PCA Algorithm
pca = PCA(n_components=13)
pca_values = pca.fit_transform(df_std)
pca_values 

# Weight values from w1 to w13
pca.components_

#Understanding variance (information carried) of each principle component
var = pca.explained_variance_ratio_
var

#calculating Cumulative Variance
var_net = np.cumsum(np.round(var,decimals = 4)*100)
var_net

#Plotting the variance of each component
plt.plot(var_net, color="blue")

#Selecting 1st 6 Principle Components which inclused 85% of the information
df_final = pd.concat([pd.DataFrame(pca_values[:, 0:6], columns= ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6']), data['Type']], axis=1)
df_final

# Normalizing the dataframe having 6 Principal components for clustering
from sklearn.preprocessing import MinMaxScaler
norm = MinMaxScaler()
df_norm = pd.DataFrame(norm.fit_transform(df_final.iloc[:, :6]))
df_norm.head(10)


# **`Clustering Method [1]  :  Hierarchical Clustering`**

import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import linkage 
h_clust = linkage(df_norm, method= "average", metric= "euclidean")
plt.figure(figsize=(15, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Euclidean_Distance')
sch.dendrogram(h_clust)
plt.show()

# Clustering of Observations
from sklearn.cluster import AgglomerativeClustering 
h_result = AgglomerativeClustering(n_clusters=5, linkage='average',affinity = "euclidean").fit(df_norm) 
cluster_labels=pd.Series(h_result.labels_)
df_final['Clusters_H']= cluster_labels     
df_final

df_final.iloc[:, :].groupby(df_final['Clusters_H']).mean()

sns.countplot(x = 'Clusters_H', data= df_final)


# **`Clustering Method [2]  :  K-Means Clustering`**

from sklearn.cluster import KMeans
fig = plt.figure(figsize=(10, 8))
WCSS = []
for i in range(1, 12):
    clf = KMeans(n_clusters=i)
    clf.fit(df_norm)
    WCSS.append(clf.inertia_)               # inertia is another name for WCSS
plt.plot(range(1, 12), WCSS)
plt.title('The Elbow Method')
plt.ylabel('WCSS')
plt.xlabel('Number of Clusters')
plt.show()

# slope is getting constant at aprroximate number of clusters = 4
k_wine = KMeans(n_clusters= 5)
y_kmeans = k_wine.fit_predict(df_norm)
y_kmeans

df_final['Clusters_K'] = pd.Series(y_kmeans) 
df_final.head(10)

df_final.iloc[:, :].groupby(df_final.Clusters_K).mean()

df_final.plot(x = "PC1", y = "PC2", c = k_wine.labels_, kind = 'scatter', s= 25, cmap = plt.cm.coolwarm)

sns.countplot(x = 'Clusters_K', data= df_final)

