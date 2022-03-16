#Sci-Kit Learn
#KNN

#Görev : Yeni bir botanist bir İris çiçeğinin hangi türde olduğunu bilmek istiyor
       #Sistemin verilerini biz vereceğimiz için bu bir Supervised öğrenim olacaktır
            
#Veriyi Tanıma - Veriyi Parçalama - Veriyi Görselleştirme - Model Oluşturma - Yeni Veri Tahmin Etme


from sklearn.datasets import load_iris
iris = load_iris()

#VERİYİ TANIMA

print(iris.keys()) #Anahtar kelimeleri gösterir

print(iris.DESCR) #Description eder

print(iris['target_names']) #Tahmin edilmek istenen türler buradadır
print(iris['feature_names']) #Niteliklerin özelliklerini gösterir

#VERİYİ PARÇALAMA

#Modelin doğru çalışıp çalışmadığını bilmek amacıyla veri 2 parçaya bölünür
           # 1-(Training Model) : Makine öğrenmesi modelini kurmak için kurulur
           # 2-(Test Model) : Modelin nasıl çalıştığını değerlendirmek amacıyla kullanılır 

#Kullanacağımız fonksiyon ilk önce satırları karıştırır daha sonra %70'ini Training için, %30'ini ise Test için kullanır
           # Veri X harfi ile , etiket ise y harfi ile gösterilir   f(X) = y
                                                                    #X girdi = Y çıktı
                                                                    
from sklearn.model_selection import train_test_split

X_training , X_test , y_training , y_test = train_test_split(iris['data'] , iris['target'] , random_state=0)
#Veriyi karıştırır 



#VERİ ÖN İNCELEME:
    
    #Veriyi incelemenin en iyi yolu görselleştirmedir.
    #Saçılım grafiği

import pandas as pd

iris_df = pd.DataFrame(X_training, columns = iris.feature_names)

#Pandas'ın Scatter Matrix isminde ikili grafikleştirme özelliği olan bir modül vardır

from pandas.plotting import scatter_matrix

#Grafikleri satır aralarında görmek için

scatter_matrix(iris_df , c = y_training , figsize = (15 , 15)  , marker = "o")


#MODELİ KURMA (CLASSIFICATION)

        #Eğitim verisi ile yapılır

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 2)

knn.fit(X_training , y_training)

#Tahmin yapma
import numpy as np
X_newtraining = np.array([[5,3.9,1,0.2]])

predict = knn.predict(X_newtraining)


print(predict)
print('Tahmin türü:' , iris['target_names'][predict])

#MODELİN PERFORMANSI

y_tahmin = knn.predict(X_test)

print(y_tahmin)

#Karşılaştırma işlemini numpy'da ki mean() ile yapabiliyoruz

print(np.mean(y_tahmin == y_test))


    
    
    
    
    
    
    
    