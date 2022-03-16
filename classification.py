#First of all StackOverFlow and codebasics 
    #Real Estate Price Prediction Project 


import pandas as pd 
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

#rcParams for scale our graphic that we had
matplotlib.rcParams["figure.figsize"] = (20, 10)

df1 = pd.read_csv(r'C:\Users\doguy\Desktop\Code\bengaluru_house_prices.csv')

#We gonna group each of area types

print(df1.groupby('area_type')['area_type'].agg('count'))

#Then drop some columns that useless

df2 = df1.drop(['area_type' , 'society', 'balcony', 'availability'] , axis = 'columns')
print(df2.shape)
print(df2.head())

#Handlining any values

df2.isnull().sum() #Number of rows where particular coumn value is any

df3 = df2.dropna() #Drop all any values
df3.isnull().sum()

#When you look dataset, size columns has same values but it has different name (Bedroom = BHK)
 #Then we'll create a new column to combine each of them.
 
df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0])) #IMPORTANT!

#[TR] x.split diyerek BHK'den öncesini ayırdık ve 0 diyerek de içine apply'ladık
#[EN] We created a new column and we combined each of size values with .apply function and lambda function

print(df3['bhk']) #Completed

def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
    
df4 = df3.copy() #We copied that
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)

#Feature Engineering 

df5 = df4.copy()
#We'll make outlier cleaning...

df5["price_per_sqft"] = df5['price']*100000/df5['total_sqft'] #MATH
print(df5.head())

print(len(df5.location.unique()))
#[TR] Location'u dummy hale getireceğiz ama çok fazla sütun olacak
     #o yüzden bunları bir temizlemek yani fazlalıkları atmak gerekecek
     
df5.location = df5.location.apply(lambda x: x.strip()) #İf we have any extra space, we'll delete it.

location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending = False) #We'll collect as group 
print(location_stats)
#5:30

location_stats_less_than_10 = location_stats[location_stats<=10]
print(location_stats_less_than_10)

df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x )
len(df5.location.unique())

df5[df5.total_sqft/df5.bhk<300].head()
#When you look at this data: 600 sqft has 8 bedroom, that's anomilies, so we'll make
#Outline Cleaner

df6 =  df5[~(df5.total_sqft/df5.bhk<300)] #A WAY OF OUTLINE CLEANER 
df6.shape

#Let's check bathroom feature
df6.bath.unique()
df6[df6.bath>10]

#Example: Our manager business says: If an apartment has bath more than number of bedrooms, remove it. 

                    # plt.hist(df6.bath, rdwith=0.8)
                    # plt.xlabel("Number of bathrooms")
                    # plt.ylabel("Count")

df7 = df6[df6.bath<df6.bhk+2] #Task 

df8 = df7.drop(['size','price_per_sqft'], axis = 'columns') #Drop

dummies = pd.get_dummies(df8.location) #Dummies

df8 = pd.concat([df8,dummies], axis = 'columns') #Concat

df9 = df8.drop('location', axis='columns') #Drop location columns
df9.head()

new_dataframe = df9.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

                    #MODEL
X = new_dataframe.drop('price', axis = 'columns')
y = new_dataframe.price

                    #Train-Test-Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 10)

                    #Cross Validation
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
cv = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state=0)
cross_val_score(LinearRegression(), X,y, cv=cv )
# [0.48873113,  0.49706053, -0.05755506,  0.26256965,  0.48752686]

                    #GridSearchCV - Dict of Algorithms
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearch_cv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['squared_error','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }

                 #
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
            })
    return pd.DataFrame(scores,columns=['model','best_params','best_score'])
print(find_best_model_using_gridsearch_cv(X,y))   
                 #

                    #Module That I Choosed
from sklearn.linear_model import LinearRegression #LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)

                    #Predictor Function
def predict_price(location,sqft,bath,bhk): 
    loc_index  = np.where(X.columns==location)[0][0]
    
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >=0:
        x[loc_index] = 1
    return lr_clf.predict([x])[0]

predict_price('1st Phase JP Nagar',1000, 2, 2)

                        #Thanks "Normalized Nerd YouTube" Channel!

                    # from xgboost import XGBRegressor
                    # xgb_model = XGBRegressor(random_state=0)

                    # search_space = {
                    #     "n_estimators" : [100, 200, 500],
                    #     "max_depth" : [3,6,9],
                    #     "gamma" : [0.01, 0.1],
                    #     "learning_rate" : [0.001,0.01,0.1,1]
                    #     }

                    # from sklearn.model_selection import GridSearchCV
                    # #Make a GridSearchCV object
                    # GS = GridSearchCV(estimator = xgb_model,
                    #                   param_grid = search_space,
                    #                   scoring = ["r2", "neg_root_mean_squared_error"],
                    #                   refit = "r2",
                    #                   cv = 5,
                    #                   verbose = 4)

                        # GS.fit(X_train, y_train)

                    #Save
import pickle
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)
    
import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
    }
with open("columns.json","w") as f:
    f.write(json.dumps(columns))