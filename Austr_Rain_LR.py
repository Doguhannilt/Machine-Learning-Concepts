import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#Predict RainTomorrow (0 = No , 1 = Yes)

data = pd.read_csv(r'C:\Users\Asus\Desktop\AUST.RAIN\weatherAUS.csv')

data = data.drop(["Date" , "Location" , "Evaporation" , "Sunshine" , "WindGustDir", "WindGustSpeed" , "WindDir9am" , "WindDir3pm"] , axis = 1)

# print(data.isnull().sum()) #You can see all NaN values

data.dropna(inplace=True) #You can delete all NaN values

Rain_Today = pd.get_dummies(data["RainToday"] , drop_first = True , prefix='Rain_Today') #Prefix

Rain_Tomorrow = pd.get_dummies(data["RainTomorrow"] , drop_first = True) #We Get Dummies.

data = data.drop(["RainToday" , "RainTomorrow"] , axis = 1) #Drop main columns

data = pd.concat([data , Rain_Today , Rain_Tomorrow] , axis = 'columns') #Concantenate

X = data.drop("Yes" , axis = 1)  
y = data.Yes

X_train, X_test, y_train, y_test = train_test_split(X , y , test_size = 0.2) 

Reg = LogisticRegression(max_iter = 1000) #Max iteration = 1000

Reg.fit(X_train , y_train)

y_pred = Reg.predict([[53.2 , 53.2 , 20.2 , 23.0 , 50.0 , 89.0 , 58.0 , 1204.8 , 1201.5 , 9.0 , 5.0 , 25.7 , 33.0 , 1]])

if Reg.predict == "0":
    print("it won't rain ")
else:
    print("it will rain")
    
print("""Predict : {} 
      
      Coef : {}
      
      İntercept : {}""".format( y_pred , Reg.coef_ , Reg.intercept_)) #M * X + B

data.to_csv("Austr_Rain_LR.csv")
      

      

    

    

    
    