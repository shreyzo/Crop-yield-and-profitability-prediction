# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 19:47:27 2020

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
datainput = pd.read_csv("datafile.csv", delimiter = ',')

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
               'Cost of Cultivation (`/Hectare) C2','Cost of Production (`/Quintal) C2', 'Support price']].values

       



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

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x3 = sc.fit_transform(x3)
#Splitting
X_train, X_test, y_train, y_test = train_test_split(x3, y, test_size=0.3, random_state=3)



#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier 
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
#Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor 
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


print("R2 score =", round(metrics.r2_score(y_test, y_pred), 2))




from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test, y_pred)
from sklearn.metrics import precision_recall_fscore_support as score

from sklearn.metrics import f1_score
f1_score(y_test, y_pred)

precision, recall, fscore, support = score(y_test, y_pred)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

##########################################
#Logitic
from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state = 0) 
classifier.fit(X_train, y_train) 
y_pred = classifier.predict(X_test) 
matrixForLR = confusion_matrix(y_test, y_pred)
##########################################



from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors=7) 
  
knn.fit(X_train, y_train) 
  
# Predict on dataset which model has not seen before 
print(knn.predict(X_test)) 
print(knn.score(X_test, y_test)) 



from sklearn.cluster import KMeans
wcss = []
for i in range(1, 30):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(x3)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 30), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(x3)