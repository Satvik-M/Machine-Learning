import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

"""## Training the Linear Regression model on the whole dataset"""

from sklearn.linear_model import LinearRegression
lin_regression=LinearRegression()
lin_regression.fit(X,y)

"""## Training the Polynomial Regression model on the whole dataset"""

from sklearn.preprocessing import PolynomialFeatures
pf=PolynomialFeatures(degree=4)
X_new=pf.fit_transform(X)
lin_regression2=LinearRegression()
lin_regression2.fit(X_new,y)

"""## Visualising the Linear Regression results"""

plt.scatter(X,y)
plt.plot(X,lin_regression.predict(X),color="red")

"""## Visualising the Polynomial Regression results"""

plt.scatter(X,y)
plt.plot(X,lin_regression2.predict(X_new),color="red")

"""## Visualising the Polynomial Regression results (for higher resolution and smoother curve)"""

x_axis=np.arange(min(X),max(X),0.01)
x_axis=x_axis.reshape(len(x_axis),1)
plt.scatter(x_axis,lin_regression2.predict(pf.transform(x_axis)))
plt.scatter(X,y,color="red")

"""## Predicting a new result with Linear Regression"""

lin_regression.predict([[6.5]])

"""## Predicting a new result with Polynomial Regression"""

lin_regression2.predict(pf.transform([[6.5]]))

