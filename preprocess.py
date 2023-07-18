import numpy as np # works well with arrays
import matplotlib.pyplot as plt # plots charts
import pandas as pd # pandas will import data

# import the dataset
dataset = pd.read_csv('./data.csv')

# create matrix of features (indepdendent) and dependent variable vector
# features are columns which we will use to predict the dependent variable

# x is matrix of features
# get all the rows, and all the columns except the last one
x = dataset.iloc[:, :-1].values

# y is the vector of depdendent variables
# get all the rows for the last column
y = dataset.iloc[:, -1].values
print(y)