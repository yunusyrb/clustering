import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veri=pd.read_csv("musteriler.csv")

x=veri.iloc[:, 3:].values

from sklearn.cluster import KMeans
km=KMeans(n_clusters=3,init='k-means++')
km.fit(x)
sonuclar=[]
print(km.cluster_centers_)
for i in range(1,11):
    km=KMeans(n_clusters=i,init='k-means++',random_state=123)
    km.fit(x)
    sonuclar.append(km.inertia_)

plt.plot(range(1,11),sonuclar)
plt.show()