import numpy as np # works well with arrays
import matplotlib.pyplot as plt # plots charts
import pandas as pd # pandas will import data
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# Fill in missing values. Some rows have missing salaries and missing ages. Replace with the mean from column
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform( x[:, 1:3])

# Encode catagorical data
# Encode independent variables
# Instead of categorizing Country as numerical, which can be mistaken as numerical order, we will create a vector to identify the country.
# We will create three new for the 3 unique values in country column, aka one hot encode the country column
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# Encode the dependent variables
# We need to transform the "yes" and "no" values to numerical values
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data into Training Set and Test set
# Do this before Feature scaling to prevent information leakage.
# 20% of data is for testing, 80% for training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


# Perform Feature scaling to prevent certain column data from dominating 
# we will use standardization
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

print(x_train)
print(x_test)