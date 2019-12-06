import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import  scipy.cluster.hierarchy as sch

veri=pd.read_csv("musteriler.csv")

x=veri.iloc[:, 3:].values

from  sklearn.cluster import  AgglomerativeClustering
ac= AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='ward')
y_tahmin = ac.fit_predict(x)

plt.scatter(x[y_tahmin== 0,0],x[y_tahmin==0,1],s=100,c='red')
plt.scatter(x[y_tahmin== 1,0],x[y_tahmin==1,1],s=100,c='blue')
plt.scatter(x[y_tahmin== 2,0],x[y_tahmin==2,1],s=100,c='green')
plt.scatter(x[y_tahmin== 3,0],x[y_tahmin==3,1],s=100,c='yellow')
plt.title("hclustering")
plt.show()

dendrogram = sch.dendrogram(sch.linkage(x,method='ward'))
plt.show()