#Diabetes 
#Linear Regression
from sklearn import datasets

diabetes = datasets.load_diabetes()

print(diabetes)

#f(X) = y

X = diabetes.data
y = diabetes.target

#Loaad X,y + Create X and y data matrices 

X,y = datasets.load_diabetes(return_X_y=True)
from sklearn.model_selection import train_test_split

X_training , x_test , y_training , y_test = train_test_split(X , y ,test_size = 0.2)

from sklearn import linear_model


model = linear_model.LinearRegression()

model.fit(X_training , y_training) #Ham hali ile test edilmiş hali

pred = model.predict(x_test)

import seaborn as sns

#y_test ile pred'i kullanacağız çünkü birisi işlenmiş birisi ise halihazırda işlenen

sns.scatterplot(y_test , pred , marker = "+")
