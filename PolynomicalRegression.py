import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv('Salary_Position.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#fitting linear regression to dataset
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

#Fitting polynomical regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4) # change degress to see the effect
X_poly=poly_reg.fit_transform(X)
# print(X_poly) # form y=cx+mx2+mx3 c=contant and m weight or coefficient
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,y)

#visualizing the linear regression results
# plt.scatter(X,y,color='red')
# plt.plot(X,lin_reg.predict(X),color='blue')
# plt.title('Linear Regression ')
# plt.xlabel('Level')
# plt.ylabel('salary')
# plt.show()

#visualizing the Polynomial regression results
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('Polynomial Regression ')
plt.xlabel('Level')
plt.ylabel('salary')
plt.show()



