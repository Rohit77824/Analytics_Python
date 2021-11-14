# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 18:22:35 2021

@author: Rohit
"""

#Multiple Regression

# Importing the libraries
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('C:/Users/Rohit/New folder/stud_reg_2.csv')
print(type(dataset))

#Data Visualization:
sns.heatmap(dataset)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,2].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

#Note: The parameter 'random_state' is used to randomly bifurcate the dataset into training &
#testing datasets. 


# Fitting Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Calculating the coefficients:
print(regressor.coef_)

#Calculating the intercept:
print(regressor.intercept_)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Accuracy of the model

#Calculating the r squared value:
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)

#Create a DataFrame
df1 = {'Actual Applicants':y_test,
'Predicted Applicants':y_pred}
df1 = pd.DataFrame(df1,columns=['Actual Applicants','Predicted Applicants'])
print(df1)

# Visualising the predicted results
import matplotlib.pyplot as plt
line_chart1 = plt.plot(y_pred,X_test, '--',c='green')
line_chart2 = plt.plot(y_test,X_test, ':', c='red')
plt.show()
#------------------------------

