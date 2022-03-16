import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
pd = pd.read_csv(r'C:\Users\Asus\Desktop\Data\Yenii.csv'  , index_col=False )
X = pd[['per capita income (US$)']]
y = pd['year']
X_train, X_test,y_train, y_test = train_test_split(X , y , test_size = 0.1 , random_state = 42)
reg = linear_model.LinearRegression()
reg.fit(X_train ,y_train)
pred = reg.predict([[2000]])
print("""Predict : {} 
      Coef : {}
      İntercept : {}""".format( pred , reg.coef_ , reg.intercept_))

#M * X + B 
    #M : COEF
    #X : PRED
    #B : INTERCEPT
    
#Save Model Using Pickle

import pickle

with open('model_pickle' , 'wb') as f:
    pickle.dump('model' , f)
    
#Load Model Using Pickle

with open('Model_pickle' , 'rb') as f:
    dm = pickle.load(f)
    
