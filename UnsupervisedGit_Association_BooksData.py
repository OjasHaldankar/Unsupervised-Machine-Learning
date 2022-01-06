#!/usr/bin/env python
# coding: utf-8

# # `Unsupervised ML  :  Association Rules`

# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori,association_rules

# Importing the dataset
df = pd.read_csv('book_assignment.csv')
df.head(10)

df.shape
df.describe()


df['ChildBks'].value_counts()
# Plotting how many users have or doesn't have 'Child Books' 
sns.countplot(x = 'ChildBks', data= df)

# Plotting how many users have or doesn't have 'Youth Books' & similar analysis could be done for all the types odf books
sns.countplot(x = 'YouthBks', data= df)


# **Case (1)  : Using Apriori with Support = 0.1 & maximum grouping length = 3**

frequent_items = apriori(df, min_support=0.1, max_len= 3,  use_colnames = True)
frequent_items

frequent_items.sort_values('support', ascending = False, inplace=True)
frequent_items.sort_values 

#Applying Association Rule Algorithm
result_1 = association_rules(frequent_items, metric="lift", min_threshold=1)
result_1.sort_values('lift',ascending = False).head(10)


# **Case (2)  : Using Apriori with Support = 0.005 & maximum grouping length = 2**

frequent_items_1 = apriori(df, min_support=0.005, max_len= 2, use_colnames = True)
frequent_items_1

result_2 = association_rules(frequent_items_1, metric="lift", min_threshold=1)
result_2.sort_values('lift',ascending = False).head(10)


# **Case (3)  : Using Apriori with Support = 0.005 & maximum grouping length = 3**

frequent_items_2 = apriori(df, min_support=0.005, max_len= 3, use_colnames = True)
frequent_items_2

result_3 = association_rules(frequent_items_2, metric="lift", min_threshold=1)
result_3.sort_values('lift',ascending = False).head(10)


# **Selecting 'result_3' Association Model as it is having highest 'lift ratio' among all 3 models with support = 0.005 & maximum grouping length = 3**


# Visualisation using Plots
plt.figure(figsize=(10,7))
plt.scatter(result_3['support'], result_3['lift'])

plt.figure(figsize=(10,7))
plt.scatter(result_3['confidence'], result_3['lift'])

