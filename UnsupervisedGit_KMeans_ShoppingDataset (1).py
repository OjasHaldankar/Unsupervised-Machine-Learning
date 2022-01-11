#!/usr/bin/env python
# coding: utf-8


# ## `Unsupvervised ML  :  K Means Clustering` 


# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans 

# Importing the dataset
df = pd.read_csv('Shopping_KM.csv')
df.head(10)

df.describe()


# Compairing how many males & females have done shopping
df['CustomerGender'].value_counts()

# Plotting the gender ratio
sns.countplot(x= 'CustomerGender', data= df)

# Checking cities where shopping has been done
df['CustomerCity'].value_counts()

# Plotting the city results
sns.countplot(x= 'CustomerCity', data= df)

# Customer age analysis
plt.hist(x= 'CustomerAge', data= df)

# Spending score analysis
plt.hist(x= 'SpendingScore', data= df)

# Checking for Customer City vs. Spending Score
plt.scatter('CustomerCity','SpendingScore', data= df)


# As 'CustomerCityId' columns is already present in the data, dropping 'CustomerCity' column & also the final column
df = df.drop(df.columns[[3, 8]], axis= 1)
df.head()

# Label Encoding the Customer Gender column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['CustomerGender_le'] = le.fit_transform(df['CustomerGender'])
df = df.drop(df.columns[[1]], axis = 1)
df.head(10)

# Checking for null values
df.isna().sum()


# **Initialize K-Means Clustering Algorithm**

# Normalizing the data to eliminate the effect of measuring units
from sklearn.preprocessing import MinMaxScaler
norm = MinMaxScaler()
df_norm = pd.DataFrame(norm.fit_transform(df))
df_norm


fig = plt.figure(figsize=(10, 8))
WCSS = []
for i in range(1, 10):
    clf = KMeans(n_clusters=i)
    clf.fit(df_norm)
    WCSS.append(clf.inertia_)               # inertia is another name for WCSS
plt.plot(range(1, 10), WCSS)
plt.title('The Elbow Method')
plt.ylabel('WCSS')
plt.xlabel('Number of Clusters')
plt.show()

# Selecting optimum number of clusters = 6 based on Elbow Method
df_k = KMeans(n_clusters= 6)
y_kmeans = df_k.fit_predict(df_norm)
y_kmeans


# Adding the Cluster column
df['Clusters_K'] = pd.Series(y_kmeans) 
df.head(10)

df.groupby(df['Clusters_K']).mean()

