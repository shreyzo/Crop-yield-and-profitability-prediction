# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 22:07:59 2020

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import utils

###############################################################################
#from google.colab import files
#uploaded = files.upload()

#datainput = pd.read_csv("crop_production.csv", deli)
datainput = pd.read_csv("crop_production.csv", delimiter = ',')
datainput = datainput.dropna()
#preprocessing
Production = (datainput.iloc[:,6]).values
Production=Production.reshape(242361,1)
#Profit = Profit.reshape(49,1)
#Profitcopy = (datainput.iloc[:,5]*datainput.iloc[:,6]-(datainput.iloc[:,2]+datainput.iloc[:,3]+(datainput.iloc[:,5]*datainput.iloc[:,4]))).values

X = datainput[['State_Name', 'District_Name', 'Crop_Year', 
               'Season','Crop', 'Area']].values

#label encoder to categorical data 
labelencoder_X = preprocessing.LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:, 0])
X[:,1] = labelencoder_X.fit_transform(X[:, 1])
X[:,3] = labelencoder_X.fit_transform(X[:, 3])
X[:,4] = labelencoder_X.fit_transform(X[:, 4])


X = X.astype(float)
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(),[0])], remainder='passthrough')   
x2 = columnTransformer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Production, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor()
clf.fit(X_train, y_train)
from sklearn.metrics import mean_absolute_error

y_pred = clf.predict(X_test)

maerandom = mean_absolute_error(y_test, y_pred)
maerandom