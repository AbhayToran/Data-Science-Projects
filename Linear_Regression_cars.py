# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 15:18:07 2025

@author: ABHAY
"""

#Predicting the prices of pre-owned cars..
#To solve this this probleam we are going to use Linear Regression model

#importing the libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns

os.chdir("C:/Users/ABHAY/Downloads")

#Reading the files
cars_data = pd.read_csv("cars_sampled.csv")
#there are total of 50,001 records in 19 columns

#lets make a new deep copy of cars_data
cars_data1 = cars_data.copy(deep=True)

#Lets see the structure of the data
cars_data1.info()
#there are 6 int dtype column and 13 object dtype column

#Now lets get the summary of the data
cars_data.describe() 
#This changes data to 3 decimal places
pd.set_option("display.float_format",lambda x: "%.3f" %x)

#Dropping the unwanted columns..
col_drop = ["name","dateCrawled","dateCreated","postalCode","lastSeen"]
cars_data1 = cars_data1.drop(columns=col_drop,axis=1)  
#Total of 5 columns removed , now only 14 columns left..

#Removing dulpicate values
cars_data1.drop_duplicates(keep="first",inplace = True)
#Total of 470 duplicated values removed

#number of null values in each column
cars_data1.isna().sum()
#we can clearly see that there are nan values..the number of nan values are:-
#vehicleType - 5152
#gearbox - 2765
#model - 2730
#fuelType - 4467
#notRepairedDamage - 9640

#now lets explore Variable yearOfRegistration
year_wise_count = cars_data1["yearOfRegistration"].value_counts().sort_index()
sns.regplot(x="yearOfRegistration",y= 'price',scatter = True,data = cars_data1)
#we can see the outliers that year of registration starts from 1000 - 9999 years

#now lets explore Variable price
price_count = cars_data1["price"].value_counts().sort_index()
sns.displot(cars_data1["price"])
cars_data1["price"].describe()
sns.boxplot(y=cars_data1["price"])

#Now lets explore the powerPS
power_count = cars_data1["powerPS"].value_counts().sort_index()
sns.distplot(cars_data1["powerPS"])
cars_data1["powerPS"].describe()
sns.boxplot(y=cars_data1["powerPS"])

#As we can see there is vast range.. so we will limit the range
cars_data1 = cars_data1[(cars_data1.yearOfRegistration<=2025) & 
                        (cars_data1.yearOfRegistration>=1980) &
                        (cars_data1.price >=100) & 
                        (cars_data1.price <= 150000) &
                        (cars_data1.powerPS >=10) &
                        (cars_data1.powerPS <= 500)]
#now we can see that only 42417 records left

#changing the month of registration
cars_data1["monthOfRegistration"]/=12
#creating new variable Age
cars_data1["Age"] = (2025-cars_data1["yearOfRegistration"])+cars_data1["monthOfRegistration"]
cars_data1["Age"].describe()

#now we will drop year and month of registration
cars_data1 = cars_data1.drop(columns=["yearOfRegistration","monthOfRegistration"]
                             ,axis = 1)
#now we left with only 13 columns

# now visualizing the parameter
#Age
sns.distplot(cars_data1["Age"])
sns.boxplot(cars_data1["Age"])

#price
sns.distplot(cars_data1["price"])
sns.boxplot(cars_data1["price"])

#powerPS
sns.distplot(cars_data1["powerPS"])
sns.boxplot(cars_data1["powerPS"])

#now lets see the relation b/w Age and price
sns.regplot(x="Age",y="price",data=cars_data1)

#now lets see the relation b/w powerPS and price
sns.regplot(x="powerPS",y="price",data=cars_data1)

#seller
cars_data1["seller"].value_counts()
pd.crosstab(index=cars_data1["seller"],columns='count',normalize=True)
sns.countplot(x=cars_data1['seller'],data = cars_data1)

#offerType
cars_data1['offerType'].value_counts()
sns.countplot(x='offerType',data=cars_data1)

#abtest
cars_data1['abtest'].value_counts()
pd.crosstab(index = cars_data1['abtest'], columns='count',normalize=True)
sns.boxplot(x='abtest',y='price',data=cars_data1)
sns.countplot(x='abtest',data=cars_data1)
#we can see that for every abtest there is 50-50 distribution of price
#it concludes that does not affect the price

#vehicleType
cars_data1["vehicleType"].value_counts()
pd.crosstab(index = cars_data1['vehicleType'], columns = 'count',normalize=True)
sns.countplot(x='vehicleType',data=cars_data1)
sns.boxplot(x='vehicleType',y='price',data=cars_data1)
#there are toatl 8 types of vehicle
#and vehicleType affect the price

#gearbox
cars_data1['gearbox'].value_counts()
pd.crosstab(index = cars_data1['gearbox'], columns='count',normalize=True)
sns.boxplot(x='gearbox',y='price',data=cars_data1)
sns.countplot(x='gearbox',data = cars_data1)
#price varies on the basis of gearbox

#model
cars_data1['model'].value_counts()
pd.crosstab(index = cars_data1['model'], columns='count',normalize=True)
sns.countplot(x='model',data=cars_data1)
sns.boxplot(x='model',y='price',data=cars_data1)

#kilometer
cars_data1['kilometer'].value_counts().sort_index()
pd.crosstab(index = cars_data1['kilometer'], columns = 'count',normalize=True)
sns.boxplot(x='kilometer',y='price',data=cars_data1)
cars_data1.describe()
sns.displot(cars_data1['kilometer'],bins=8,kde=False)
sns.regplot(x='kilometer',y='price',scatter = True,data = cars_data1)

#fuelType
cars_data1['fuelType'].value_counts()
pd.crosstab(index = cars_data1['fuelType'],columns = 'count',normalize=True)
sns.countplot(x='fuelType',data = cars_data1)
sns.boxplot(x='fuelType',y='price',data=cars_data1)
#fuelType varies the price..

#brand
cars_data1['brand'].value_counts()
pd.crosstab(index = cars_data1['brand'],columns='count',normalize=True)
sns.countplot(x='brand',data=cars_data1)
sns.boxplot(x='brand',y='price',data=cars_data1)
#on the basis of brand the price alters

#notRepairedDamage
cars_data1['notRepairedDamage'].value_counts()
#no -> represents (cars damaged but rectified)
#yes -> represents (cars damaged but not rectified)
pd.crosstab(index = cars_data1['notRepairedDamage'], columns='count',normalize=True)
sns.countplot(x='notRepairedDamage',data=cars_data1)
sns.boxplot(x='notRepairedDamage',y='price',data=cars_data1)
#prices vary on the basis of damage

#Now lets remove the insignificant variables..
cars_data1 = cars_data1.drop(columns=['seller','offerType','abtest'],axis=1)
#now left with only 10 columns..

#now lets see the correlation
cars_select1 = cars_data1.select_dtypes(exclude=[object])
correlation_cars_data1 = cars_select1.corr()
cars_select1.corr()['price'].abs().sort_values(ascending=False)[1:]

#Omitting missing values..
cars_data_omit = cars_data1.dropna(axis=0)
#now we get only 32765 records

#converting categorical data into the dummy variable
cars_data_omit = pd.get_dummies(data=cars_data_omit,drop_first=True)
#after converting we got 300 columns

#lets import the necessary library
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

#Model building with ommited data
#x1 is input data , y1 is the output data
x1 = cars_data_omit.drop(['price'],axis='columns',inplace=False)
y1 = cars_data_omit['price']

#plotting the variable price
prices = pd.DataFrame({"1.Before":y1,"2.After":np.log(y1)})

#transforming the price as the logarithmic value
y1 = np.log(y1)

#spliting the train and test data
train_x,test_x,train_y,test_y = train_test_split(x1,y1,test_size=0.3,random_state=3)
#we can see that train data - 22872, test data - 9803

#BaseLine model for ommited value
#finding the mean for the test data
base_pred = np.mean(test_y) 
#8.233
y_pred_base = np.full_like(test_y, base_pred)  
# array filled with mean value


#finding the RMSE
base_rmse = np.sqrt(mean_squared_error(test_y, y_pred_base))
print(base_rmse)
#RMSE value is 1.143

#Linear regression with ommited data
#setting intercept as true

lgr = LinearRegression(fit_intercept=True) 

#model
model_lin1 = lgr.fit(train_x,train_y)

#predicting model on test data
cars_linear_predction = lgr.predict(test_x)

#computing MSE and RMSE
linear_mse = mean_squared_error(test_y,cars_linear_predction)
linear_rmse = np.sqrt(linear_mse)
print(linear_rmse)
#RMSE - 0.4863 less than earlier one

#R squared value
r2_linear_test1 = model_lin1.score(test_x,test_y)
r2_linear_train1 = model_lin1.score(train_x,train_y)
print(r2_linear_test1) #0.8191418121276296
print(r2_linear_train1) #0.8193013903709867

#Random forest with omitted data
rf = RandomForestRegressor(n_estimators=100,max_features='sqrt'
                           ,max_depth=100,min_samples_split=10,
                          min_samples_leaf=4,random_state=1)

#model
model_rf1 = rf.fit(train_x,train_y)

#predictig model on test data
cars_rf_prediction = rf.predict(test_x)

#Computing RMSE and MSE
rf_mse = mean_squared_error(test_y,cars_rf_prediction)
rf_rmse = np.sqrt(rf_mse)
print(rf_rmse)
#0.4877 greater than linear regression

#R squared value
r2_rf_test1 = model_rf1.score(test_x,test_y)
r2_rf_train1 = model_rf1.score(train_x,train_y)
print(r2_rf_test1) #0.8181268109723364
print(r2_rf_train1) #0.8314901997986568
