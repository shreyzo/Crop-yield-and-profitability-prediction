# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 11:25:00 2020

@author: Shreya
"""
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn import utils

###############################################################################
datainput = pd.read_csv("datafile.csv", delimiter = ',')

#preprocessing
X = datainput[['Crop', 'State', 'Cost of Cultivation (`/Hectare) A2+FL', 'Cost of Cultivation (`/Hectare) C2','Cost of Production (`/Quintal) C2' ]].values



#label encoder to categorical data 
labelencoder_X = preprocessing.LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:, 0])
X[:,1] = labelencoder_X.fit_transform(X[:, 1])

#One hot encoder 
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(),[0])], remainder='passthrough')   
x2 = np.array(columnTransformer.fit_transform(X), dtype = np.float) 

columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(),[10])], remainder='passthrough')  
x3 = np.array(columnTransformer.fit_transform(x2), dtype = np.float) 

#output col in y 
y = datainput.iloc[:,-1]

#Splitting
X_train, X_test, y_train, y_test = train_test_split(x3, y, test_size=0.3, random_state=3)

#Decision Tree Regressor
regressor = DecisionTreeRegressor(max_depth = 4)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)



