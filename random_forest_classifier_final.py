# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 12:29:24 2020

@author: aakanksha
"""

import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import utils

###############################################################################
datainput = pd.read_csv("data_final.csv", delimiter = ',')

#preprocessing
Profit = (datainput.iloc[:,5]*datainput.iloc[:,6]-(datainput.iloc[:,2]+datainput.iloc[:,3]+(datainput.iloc[:,5]*datainput.iloc[:,4]))).values
Profit = Profit.reshape(49,1)
Profitcopy = (datainput.iloc[:,5]*datainput.iloc[:,6]-(datainput.iloc[:,2]+datainput.iloc[:,3]+(datainput.iloc[:,5]*datainput.iloc[:,4]))).values

for i in range (0,49):
    if Profit[i][0]>0:
        Profit[i][0] = 1
    else:
        Profit[i][0] = 0
X = datainput[['Crop', 'State', 'Cost of Cultivation (`/Hectare) A2+FL', 
               'Cost of Cultivation (`/Hectare) C2','Cost of Production (`/Quintal) C2', 'Support price', 'Yield (Quintal/ Hectare) ']].values

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
y = Profit


X_train, X_test, y_train, y_test = train_test_split(x3, y, test_size=0.25, random_state=42)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 100, random_state = 42)
classifier.fit(X_train, y_train)

# Predicting a new result
y_pred = classifier.predict(X_test)
y_pred

print("R2 score =", round(metrics.r2_score(y_test, y_pred), 2))




from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test, y_pred)