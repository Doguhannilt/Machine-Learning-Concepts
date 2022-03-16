#Support Vector Machines 
import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()

#https://www.youtube.com/watch?v=FB5EdxAGxQg&list=PLeo1K3hjS3uvCeTYTeyfe0-rN5r8zn9rw&index=11


from sklearn.model_selection import train_test_split

df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['target'] = iris.target
X = df.drop(['target'] , axis = 1)
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


from sklearn.svm import SVC
model = SVC()   #Regularization C = 1.0 , gamma = "auto" , kernel = "rbf" ##Default Mode
model.fit(X_train, y_train) 

print(model.score(X_test, y_test))
print(model.predict([[4.8,3.0,1.5,0.3]]))