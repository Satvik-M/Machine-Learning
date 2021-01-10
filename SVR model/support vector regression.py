

## Importing the libraries
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""## Importing the dataset"""

data=pd.read_csv("Position_Salaries.csv")
X=data.iloc[:,1:-1].values
y=data.iloc[:,-1].values

X

y=y.reshape(len(y),1)
y

"""## Feature Scaling"""

from sklearn.preprocessing import StandardScaler
X_sc=StandardScaler()
y_sc=StandardScaler()
X=X_sc.fit_transform(X)
y=y_sc.fit_transform(y)

"""## Training the SVR model on the whole dataset"""

from sklearn.svm import SVR
regressor=SVR()
regressor.fit(X,y)

"""## Predicting a new result"""

y_sc.inverse_transform(regressor.predict(X_sc.transform([[6.5]])))

"""## Visualising the SVR results"""

plt.scatter(X_sc.inverse_transform(X),y_sc.inverse_transform(y),color="red")
plt.plot(X_sc.inverse_transform(X),y_sc.inverse_transform(regressor.predict(X)))
plt.show()

"""## Visualising the SVR results (for higher resolution and smoother curve)"""

X_grid=np.arange(min(X_sc.inverse_transform(X)),max(X_sc.inverse_transform(X)),0.01)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X_sc.inverse_transform(X),y_sc.inverse_transform(y),color="red")
plt.plot(X_grid,y_sc.inverse_transform(regressor.predict(X_sc.transform(X_grid))))
plt.show()

