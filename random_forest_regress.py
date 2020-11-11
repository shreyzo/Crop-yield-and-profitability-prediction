# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 20:36:20 2020

@author: aakanksha
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('datafile.csv')
X = dataset[['Crop', 'State', 'Cost of Cultivation (`/Hectare) A2+FL', 'Cost of Cultivation (`/Hectare) C2','Cost of Production (`/Quintal) C2' ]].values
y = dataset.iloc[:,-1]

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
#label encoder to categorical data 
labelencoder_X = preprocessing.LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:, 0])
X[:,1] = labelencoder_X.fit_transform(X[:, 1])

#One hot encoder 
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(),[0])], remainder='passthrough')   
x2 = np.array(columnTransformer.fit_transform(X), dtype = np.float) 

columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(),[10])], remainder='passthrough')  
x3 = np.array(columnTransformer.fit_transform(x2), dtype = np.float) 

from sklearn.model_selection import train_test_split

#x3 = dataset.x3.values.reshape(-1, 1)
#y = dataset.y.values.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x3, y, test_size=0.30, random_state=42)

print(dataset.columns)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(x_train, y_train)

# Predicting a new result
y_pred = regressor.predict(x_test)
y_pred


from sklearn.metrics import r2_score,mean_squared_error
mse = mean_squared_error(y_test, y_pred)
mse
rmse = np.sqrt(mse)
rmse