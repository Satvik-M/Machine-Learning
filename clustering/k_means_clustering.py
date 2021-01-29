##Not all  the data is included for better visualisation
## Importing the libraries
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""## Importing the dataset"""

dataset=pd.read_csv("Mall_Customers.csv")
X=dataset.iloc[:,[3,4]].values

"""## Using the elbow method to find the optimal number of clusters"""

from sklearn.cluster import KMeans
wcss=[];
for i in range(1,11):
  kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
  kmeans.fit(X)
  wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.show()

"""## Training the K-Means model on the dataset"""

kmeans=KMeans(n_clusters=5,init='k-means++',random_state=42)
y_pred=kmeans.fit_predict(X)
print(y_pred)

"""## Visualising the clusters"""

plt.scatter(X[y_pred==0,0],X[y_pred==0,1],c='red',label='1')
plt.scatter(X[y_pred==1,0],X[y_pred==1,1],c='lime',label='2')
plt.scatter(X[y_pred==2,0],X[y_pred==2,1],c='cyan',label='3')
plt.scatter(X[y_pred==3,0],X[y_pred==3,1],c='brown',label='4')
plt.scatter(X[y_pred==4,0],X[y_pred==4,1],c='black',label='5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='orange',label='center')
plt.legend()

