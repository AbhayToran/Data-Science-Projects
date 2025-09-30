# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 10:24:12 2025

@author: ABHAY
"""

#importing Libraries
# Question is about giving subsidy based on the data of individual income 

import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix


#Reading file and storeing in dataframe
os.chdir("C:/Users/ABHAY/Downloads")
income_data = pd.read_csv("income(1).csv")

#Now analysing the given data (Exploratory data analysis)
#Making a copy of the data that is Deep copy rather than shallow copy
income_data1=income_data.copy(deep=True)

income_data1.info()
#from the info we can see that there are no null data.. 
#4 are int dtype and 9 are object dtype columns 
#memory usage is 3.2 MB

#useing of describe function to see all 8 parameters of int dtypes
income_data1.describe()
#we can see that maximum neither gained capital or loss capital

#now useing describe function to see the object dtype 
object_data_descibe=income_data1.describe(include="O")
# we can see that there are 
# 9 unique JobType, 16 unique EdType,etc...

#Now lets check the unique values of object dtypes
print(np.unique(income_data1["JobType"]))
print(np.unique(income_data1["EdType"]))
print(np.unique(income_data1["maritalstatus"]))
print(np.unique(income_data1["occupation"]))
print(np.unique(income_data1["relationship"]))
print(np.unique(income_data1["race"]))
print(np.unique(income_data1["gender"]))
print(np.unique(income_data1["nativecountry"]))
print(np.unique(income_data1["SalStat"]))
# we can see that there is unwanted values like "?" in JobType and occupation
# we need to change "?" to nan_values

#Now converting "?" to nan and storeing in new data frame
income_data2 = pd.read_csv("income(1).csv",na_values=[" ?"])

income_data2.info()

#Now checking the number of nan values in each column
income_data2.isnull().sum()
#After converting to nan_values we can see that there are empty values..
#JobType has 1809 na_values
#occupation has 1816 na_values

#as they both are object dtypes rather than filling we will remove rows
#at first there was 31978 rows  
income_data2 = income_data2.dropna(axis = 0)
#after dropping there are only 30162 records total of 1816 records removed...

#Lets check the correlation
numerical_data = income_data2.select_dtypes(exclude=[object])
corr_matrix = numerical_data.corr()
#from this we can get the insights that how the numerical data are related wheather strong relation or weak relation

#Now lets check the stats with object dtypes
pd.crosstab(index = income_data2["EdType"], columns= income_data2["SalStat"],
            margins = True,normalize = True)

pd.crosstab(index = income_data2["occupation"], columns= income_data2["SalStat"],
            margins = True,normalize = True)

pd.crosstab(index = income_data2["gender"], columns= income_data2["SalStat"],
            margins = True,normalize = True)

#LOGISTIC REGRESSION

#Converting less than or equal to 50,000 to 0
#and greater than 50,000 to 1
income_data2["SalStat"] = income_data2["SalStat"].map({' greater than 50,000':1, ' less than or equal to 50,000':0})

#craeting new dataframe and using get_dummies
#Converts all the data to 0 and 1 
new_data = pd.get_dummies(income_data2,drop_first=True)

#storeig all column names
#has 95 columns
coloumns_list = list(new_data.columns)
print(coloumns_list)

#Separating the input names from data
#has 94 columns
features = list(set(coloumns_list)-set(['SalStat']))

#storeing the o/p values in y
y = new_data['SalStat'].values

#storeing the values from i/p features
x = new_data[features].values

#Splitting the data into train and test
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3,random_state=0)


#Making the instance of the model
logistic = LogisticRegression()

#Fitting the values x and y
logistic.fit(train_x,train_y)

#Predicting from test data
prediction = logistic.predict(test_x)

#Now creating confusion matrix which evaluates the performance of classification model
logistic_confusion_matrix = confusion_matrix(test_y,prediction)
print(logistic_confusion_matrix)

#calculating the accuracy score
score = accuracy_score(test_y,prediction)
print(score)
#got 83.61 percentage accuracy


##############################################################################
#KNN
#importing libraries
from sklearn.neighbors import KNeighborsClassifier

#Storeing the k nearest neighbours classifiers
knn_classifier = KNeighborsClassifier(n_neighbors=5)

#Fitting thevalues for X and Y
knn_classifier.fit(train_x,train_y)

#prediction of the test values with models..
knn_prediction  = knn_classifier.predict(test_x)

#Performance metrics check
knn_confusion_matrix = confusion_matrix(test_y,knn_prediction)
print("\tPrediction values")
print("Original values \n",knn_confusion_matrix)


#calculating the accuracy
knn_accuracy_score = accuracy_score(test_y,knn_prediction)
print(knn_accuracy_score)
#we can see that Knn gives better accuracy_score
#got the score of 83.92%

#now lets draw the pair plot..
sns.pairplot(income_data2,kind="boxplot",hue="SalStat")

sns.boxplot(x="gender",y="SalStat",data=income_data2)

sns.countplot(x="gender",data=income_data2,hue="SalStat")

sns.countplot(x="JobType",data=income_data2)
