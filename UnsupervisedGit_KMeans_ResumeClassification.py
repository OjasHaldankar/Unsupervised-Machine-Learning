#!/usr/bin/env python
# coding: utf-8

# # `KMeans Clustering Algorithm  :  Unsupervised ML`


# Input Data : We have been provided with a list of skills (skillset) of candidates belonging to different profiles. This data is not labelled i.e. appropriate profiles are not mapped based on mentioned skillsets. 

# Objective : The objective is to map correct profiles based on mentioned skillset (the study/research of profile vs skills is already carried out) by using KMeans Clustering algorithm


# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
import seaborn as sns
import string
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud,STOPWORDS

# Importing the base dataset
pd.set_option("display.max_rows", None)
df = pd.read_csv("Resume Classification CSV.csv")
df

# Extracting / Considering only the 'Skills' data which is required for analysis
ds = df[['Skills']]
list = ds['Skills'].values.tolist()
list

out = []
for i in list:
    out += i.split("|")
print(out)

df1 = pd.DataFrame(out, columns= ['Skill'])

# Creating WordCloud
text = df1["Skill"]
wordcloud = WordCloud(
    width = 2400,
    height = 1200,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(text))
fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# ### `Initializing KMeans Clustering Algorithm` 

from sklearn.cluster import KMeans 

# TF - IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf_vectorizor = TfidfVectorizer(stop_words = 'english',              #tokenizer = tokenize_and_stem,
                             max_features = 20000)
tf_idf = tf_idf_vectorizor.fit_transform(ds["Skills"].values.astype('U')) 

# Assuming 6 number of cluster
true_k = 6
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=200, n_init=1, random_state= 32)
model.fit(tf_idf)
prediction = model.fit_predict(tf_idf)

md = pd.Series(prediction)             # converting numpy array into pandas series object 
ds['Clusters'] = md                    # creating a  new column and assigning it to new column 
ds

# Inspecting Cluster == 0
Input = ds.loc[ds['Clusters'] == 0]
post1 = pd.DataFrame(Input)
post1

# Inspecting Cluster == 1
Input2 = ds.loc[ds['Clusters'] == 1]
post2 = pd.DataFrame(Input2)
post2

# Inspecting Cluster == 2
Input3 = ds.loc[ds['Clusters'] == 2]
post3 = pd.DataFrame(Input3)
post3

# Inspecting Cluster == 3
Input4 = ds.loc[ds['Clusters'] == 3]
post4 = pd.DataFrame(Input4)
post4

# Inspecting Cluster == 4
Input5 = ds.loc[ds['Clusters'] == 4]
post5 = pd.DataFrame(Input5)
post5

# Inspecting Cluster == 5
Input6 = ds.loc[ds['Clusters'] == 5]
post6 = pd.DataFrame(Input6)
post6

Shape = {'Shpe of Cluster': [post1.shape, post2.shape, post3.shape, post4.shape, post5.shape, post6.shape]}
table = pd.DataFrame(Shape)
table

# Mapping 'Profiles' based on 'skillset' of individual clusters
ds['Category'] = ds['Clusters'].map({0 : 'React Js Developer', 1 : 'Workday Consultant', 2 : 'Peoplesoft FSCM', 3 : 'Peoplesoft Admin', 4 : 'SQL Developer', 5 : 'React Js Developer'})
ds

ds['Category'].value_counts()

