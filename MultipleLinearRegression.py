import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset=pd.read_csv('Data_company.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values
# print(X)
# print(y)

#Encoding the categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder_X=LabelEncoder()
X[:,3]=labelEncoder_X.fit_transform(X[:,3])

oneHotEncoder=OneHotEncoder(categorical_features=[3])
X=oneHotEncoder.fit_transform(X).toarray()
# print(X)

#avoiding dummy variable trap
X=X[:,1:]

#Splitting into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#fitting multuiple regression model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
print(regressor.coef_)
print(regressor.intercept_)

#predicting the test set result
y_pred=regressor.predict(X_test)

#building the optimal model using backward elimination
import statsmodels.formula.api as sm
import statsmodels.regression.linear_model as lm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5]]

regressor_OLS=lm.OLS(endog=y,exog=X_opt).fit() # OLS= ordinary least square
# print(regressor_OLS.summary())

#eliminating x1 from X, removimg first column
X_opt = X[:, [0, 2, 3, 4, 5]]
regressor_OLS=lm.OLS(endog=y,exog=X_opt).fit() # OLS= ordinary least square
# # print(regressor_OLS.summary())

#eliminating x2 from X, removimg second column
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS=lm.OLS(endog=y,exog=X_opt).fit() # OLS= ordinary least square
# print(regressor_OLS.summary())

#eliminating x4 from X, removimg fourth column
X_opt = X[:, [0, 3, 5]]
regressor_OLS=lm.OLS(endog=y,exog=X_opt).fit() # OLS= ordinary least square
# print(regressor_OLS.summary())

#eliminating x5 from X, removimg fifth column
X_opt = X[:, [0, 3]]
regressor_OLS=lm.OLS(endog=y,exog=X_opt).fit() # OLS= ordinary least square
print(regressor_OLS.summary())





