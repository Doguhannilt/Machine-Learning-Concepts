#K-Fold Cross Validation

#Veriye göre algoritmalarının hangisinin daha iyi olduğunu anlamaya yarar.
    #Split Methodundan farkı bunu katlamalı bir şekilde yapabilmesidir (n = n tane kadar sayı)
    
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
digits = load_digits()


from sklearn.model_selection import StratifiedKFold

#Bu stratifiedKFold classification aracına hizmet eder ve buna göre şekillenir. Daha gelişmiş halidir.

folds = StratifiedKFold(n_splits=3) #Sistemi üç defa böleceğimizi söylüyoruz  (n_splits = 3 )



#Burada score olarak alması için sistemi ayarlıyoruz
def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


scores_logistic = []
scores_svm = []   #SCORE TABLOLARINI BURAYA YÜKLEYECEĞİZ 
scores_rf = []


#Bu genel geçer bir for dönügüsü ve X ve y olarak ayırmanın farklı bir yolu
for train_index, test_index in folds.split(digits.data,digits.target):
    
    #Split'te yaptığımız gibi Digits_data burada X {Train,Test = Train_index} , y {train,test = digits.data}olarak alınıyor
    X_train, X_test, y_train, y_test = digits.data[train_index], digits.data[test_index], \
                                       digits.target[train_index], digits.target[test_index]
    scores_logistic.append(get_score(LogisticRegression(solver='liblinear',multi_class='ovr'), X_train, X_test, y_train, y_test))  
    scores_svm.append(get_score(SVC(gamma='auto'), X_train, X_test, y_train, y_test))
    scores_rf.append(get_score(RandomForestClassifier(n_estimators=40), X_train, X_test, y_train, y_test))

scores_logistic





