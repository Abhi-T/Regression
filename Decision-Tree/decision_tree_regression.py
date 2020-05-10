import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv('Salary_Position.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#fitting decision tree regression to dataset
from sklearn.tree import DecisionTreeRegressor
dec_reg=DecisionTreeRegressor(random_state=0)
dec_reg.fit(X,y)

#predict the value
y_pred=dec_reg.predict([[6]]) # input should be 2d array
print(y_pred)
#visualizing
plt.scatter(X,y,color='red')
plt.plot(X,dec_reg.predict(X),color='blue')
plt.title('Decision Tree Regressor')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#smoother curve
x_grid=np.arange(min(X), max(X), 0.1)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(X,y,color='red')
plt.plot(x_grid,dec_reg.predict(x_grid),color='blue')
plt.title('Decision Tree Regressor_1')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()