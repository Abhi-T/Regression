import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv('salary.csv')
# print(dataset.head())
X=dataset.iloc[:,:1].values
y=dataset.iloc[:,1].values

#splitting into test and training set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

#Fitting simple linear regressor to training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predicting the test result
y_pred=regressor.predict(X_test)

#visualizing the training set results
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('salary vs exp')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.show()

#visualizing the test set results
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('salary vs exp')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.show()