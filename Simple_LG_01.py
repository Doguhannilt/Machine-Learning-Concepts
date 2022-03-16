#Linear Regression

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model



data = pd.read_csv(r'C:\Users\Asus\Desktop\Data\Ecommerce Customers.csv')

print(data.columns)

X = data[[ 'Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]

y = data['Yearly Amount Spent']

X_training , y_training , x_test , y_test = train_test_split(X , y , test_size = 0.3 , random_state=42)

model = linear_model.LinearRegression()
model.fit(X_training , x_test)

y_pred = model.predict(X_training)

plt.scatter(y_pred , x_test)


