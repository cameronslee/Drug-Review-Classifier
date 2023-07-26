#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 10:15:10 2023

@author: cameron
"""

#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

df = pd.read_csv('drugLib_raw/drugLibTest_raw.tsv', sep='\t')

df.shape
df.head()
df.columns

#print(df)

#print(df.nunique(axis=0))
# Get the top 10 categories for categorical_var_3
top_categories = df['condition'].value_counts().head(10).index

# Filter the DataFrame to only include rows with one of the top categories
df_filtered = df[df['condition'].isin(top_categories)]

# Create a pivot table of the mean numerical value for each combination of the categorical variables
pivot_table = df_filtered.pivot_table(values='rating', index='sideEffects', columns=['condition'], aggfunc='mean')

# Create the heatmap using Seaborn
sns.heatmap(pivot_table, cmap='Blues', annot=True, fmt='.1f')

# Set the title and axis labels
plt.title('Relationship between side effects, condition and rating')
plt.xlabel('condition')
plt.ylabel('side effects')

# Show the plot
plt.show()




#print(df.effectiveness.unique())

#Rating Information
ratingInfo = df.describe().apply(lambda s: s.apply(lambda x: format(x, 'f')))
print(ratingInfo)
#print(df.describe().apply(lambda s: s.apply(lambda x: format(x, 'f'))))
df['rating'].plot(kind='hist', figsize=(12,6),title="Histogram of Ratings", facecolor='grey',edgecolor='black')
plt.title("Histogram of Ratings")
plt.xlabel('rating')
plt.ylabel('frequency')

# Show the plot
plt.show()

#Effectiveness
counts = df['effectiveness'].value_counts()
counts.plot(kind='bar', facecolor='grey',edgecolor='black')
plt.title("Effectiveness Bar Graph")
plt.xlabel('Effectiveness')
plt.ylabel('Count')

# Show the plot
plt.show()

#df['effectiveness'].plot(kind='bar', figsize=(12,6),title="Bar Graph of Effectiveness", facecolor='grey',edgecolor='black')

#Side Effects
counts = df['sideEffects'].value_counts()
counts.plot(kind='bar', facecolor='grey',edgecolor='black')
plt.title("Side Effects Bar Graph")
plt.xlabel('sideEffects')
plt.ylabel('Count')

# Show the plot
plt.show()

#Condition
counts = df['condition'].value_counts()

#lots of conditons, plot just top 50
top_50 = counts[:10]

top_50.plot(kind='bar', facecolor='grey',edgecolor='black')
plt.title("Condition Bar Graph: Top 10 Features")
plt.xlabel('condition')
plt.ylabel('Count')

# Show the plot
plt.show()

#Analyze relationships between features











