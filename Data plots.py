# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 18:41:59 2020

@author: user
"""

import pandas as pd
import matplotlib.pyplot as plt

###############################################################################
datainput = pd.read_csv("datafile.csv", delimiter = ',')

####################################################################
#Plot 1. Crops in percentage
Crops = datainput.iloc[:,0]
CropsCount = {} 
for crop in Crops: 
    CropsCount[crop] = CropsCount.get(crop, 0)+1
    
#extract values and keys of dict:CropsCount
labels = list(CropsCount.keys())
values = list(CropsCount.values())

plt.pie(values, labels=labels,autopct='%1.1f%%')
plt.show()

####################################################################
#Plot 2. State distribution
States = datainput.iloc[:,1]
StatesCount = {} 
for state in States: 
    StatesCount[state] = StatesCount.get(state, 0)+1
    
#extract values and keys of dict:CropsCount
labels = list(StatesCount.keys())
values = list(StatesCount.values())

plt.pie(values, labels=labels,autopct='%1.1f%%')
plt.show()  

###################################################################
#Plot 3.
