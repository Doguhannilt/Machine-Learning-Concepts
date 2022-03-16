#K-Means Cluster algoritmasının mantığı:
    
    #"K" buradaki kümeleme sayısını gösterir ve veriyi ne kadar fazla kümeleyeceğimize karar veren (Sum Squad Error)
    #teknik ise Elbow Method'dur. 

#Example : Use iris flower dataset from sklearn library and try to form clusters of flowers using petal width and length features. Drop other two features for simplicity.

from sklearn.datasets import load_iris
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df.drop(["sepal length (cm)" , "sepal width (cm)"] , axis = 1 , inplace = True )
print(df)

plt.scatter(df["petal length (cm)"],df["petal width (cm)"])
plt.xlabel('Petal Width ')
plt.ylabel('Petal Length')


km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[["petal length (cm)" , "petal width (cm)" ]])

df['cluster']=y_predicted
#print(km.cluster_centers_)

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]

plt.scatter(df1["petal length (cm)"] , df1["petal width (cm)"] , color='green')
plt.scatter(df2["petal length (cm)"], df2["petal width (cm)"], color='red')
plt.scatter(df3["petal length (cm)"] , df3["petal width (cm)"] , color ="yellow" )
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='Centroid')
plt.legend()
plt.show()

plt.savefig('KMeans_Cluster.png')

#Elbow Method

# sse = []
# k_rng = range(1,10)
# for k in k_rng:
#     km = KMeans(n_clusters=k)
#     km.fit(df[['petal length (cm)','petal width (cm)']])
#     sse.append(km.inertia_)
    
# plt.xlabel('K')
# plt.ylabel('Sum of squared error')
# plt.title('Elbow Method')
# plt.plot(k_rng,sse)

#Result : 3.0



#Min_Max Scaler

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()

# scaler.fit(df[['petal length (cm)']])
# df['petal length (cm)'] = scaler.transform(df[['petal length (cm)']])

# scaler.fit(df[['petal width (cm)']])
# df['petal width (cm)'] = scaler.transform(df[['petal width (cm)']])

#reference : https://github.com/codebasics/py/blob/master/ML/13_kmeans/13_kmeans_tutorial.ipynb

