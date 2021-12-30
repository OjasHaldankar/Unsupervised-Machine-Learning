#!/usr/bin/env python
# coding: utf-8

# # `Unsupervised Machine Learning  :  Clustering Algoritjms`

# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Importing the dataset
df = pd.read_csv("crime_data.csv")
df.head(10)

df.shape
df.describe()

#Normalizing the data to eliminate the effect of measuring units
norm = MinMaxScaler()
df_norm = pd.DataFrame(norm.fit_transform(df.iloc[:, 1:]))
df_norm


# **`Method (1)  :  H-CLUSTERING (Hierarchical Clustering)`**

# Importing Libraries
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import linkage 

h_clust = linkage(df_norm, method= "complete", metric= "euclidean")
plt.figure(figsize=(15, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Euclidean_Distance')
sch.dendrogram(h_clust)
plt.show()

# Observing the clusters
from sklearn.cluster import AgglomerativeClustering 
h_result = AgglomerativeClustering(n_clusters=5, linkage='complete',affinity = "euclidean").fit(df_norm) 

cluster_labels = pd.Series(h_result.labels_)
cluster_labels
df['Clusters_H']= cluster_labels                             # creating a  new column and assigning it to new column 
df

# Plotting the Clustering Results
sns.countplot(x = 'Clusters_H', data = df)

df.iloc[:, 1:].groupby(df['Clusters_H']).mean()


# **`Method (2)  :  K-MEANS CLUSTERING`**

from sklearn.cluster import KMeans
fig = plt.figure(figsize=(10, 8))
WCSS = []
for i in range(1, 10):
    clf = KMeans(n_clusters=i)
    clf.fit(df_norm)
    WCSS.append(clf.inertia_)                               # inertia is another name for WCSS
plt.plot(range(1, 10), WCSS)
plt.title('The Elbow Method')
plt.ylabel('WCSS')
plt.xlabel('Number of Clusters')
plt.show()

# Considering K = 5 based on Elbow Method 
K_means = KMeans(n_clusters= 5)
y_kmeans = K_means.fit_predict(df_norm)
y_kmeans

df['Clusters_K'] = pd.Series(y_kmeans) 
df.head(15)

df.iloc[:, 1:].groupby(df['Clusters_K']).mean()

df.plot(x = "Assault", y = "UrbanPop", c = K_means.labels_, kind = 'scatter', s= 20, cmap = plt.cm.coolwarm)

sns.countplot(x = 'Clusters_K', data= df)


# **`Method (3)  :  DB SCAN CLUSTERING`**

db = pd.read_csv('crime_data.csv')
db.drop(db.columns[[0]] , axis=1, inplace= True)
db.head()

array = db.values
# Standardizing the data
from sklearn.preprocessing import StandardScaler
SD = StandardScaler()
X = SD.fit_transform(array)
X

# Initialize DB Scan Model
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=1.25, min_samples=5)       
dbscan.fit(X) 

d1 = dbscan.labels_
d1

import sklearn as sk
sk.metrics.silhouette_score(X, d1)

df['Clusters_DB'] = d1
df

sns.countplot(x = 'Clusters_DB', data= df)

plt.style.use('classic') 
df.plot(x= "Rape", y ="UrbanPop", c= dbscan.labels_ ,kind="scatter",s=70 ,cmap=plt.cm.copper_r) 
plt.title('Clusters using DBScan') 

