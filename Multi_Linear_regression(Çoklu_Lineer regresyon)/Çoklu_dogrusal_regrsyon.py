# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 00:01:06 2021

@author: ahmet
"""


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt


veriler=pd.read_csv("features.csv")
print(veriler)

Yas=veriler.iloc[:,1:4].values
print(Yas)
#########Kategorik veri dönüşümü 

ulke=veriler.iloc[:,0:1].values
print(ulke)
from sklearn import  preprocessing 
le=preprocessing.LabelEncoder()
ulke[:,0]=le.fit_transform(veriler.iloc[:,0])

ohe=preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)



cinsiyet=veriler.iloc[:,-1:].values
from sklearn import  preprocessing 
le=preprocessing.LabelEncoder()
cinsiyet[:,-1]=le.fit_transform(veriler.iloc[:,-1])
ohe=preprocessing.OneHotEncoder()
cinsiyet=ohe.fit_transform(cinsiyet).toarray()

print(cinsiyet)

########DATA FRAME DONUŞUMU

sonuc_ulke=pd.DataFrame(data=ulke,index=range(22),columns=["fr","tr","us"])
print(sonuc_ulke)


sonuc_boykiloyas=pd.DataFrame(data=Yas,index=range(22),columns=["boy","kilo","yas"])
print(sonuc_boykiloyas)

sonuc_cinsiyet=pd.DataFrame(data=cinsiyet[:,:1], index=range(22),columns=["cinsiyet"])
print(sonuc_cinsiyet)


#DATA FRAME TOPLAMA 
s=pd.concat([sonuc_ulke,sonuc_boykiloyas], axis=1)
s2=pd.concat([s,sonuc_cinsiyet],axis=1)
print(s2)

##############verileri ölçekleme 




#######VERİLERİN EĞİTİM VE TEST İÇİN BÖLÜNMESİ 

from sklearn.model_selection import train_test_split
x_test,x_train,y_test,y_train = train_test_split(s,sonuc_cinsiyet,test_size=0.33,random_state=0)



from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(x_train,y_train)
y_pred=regression.predict(x_test)

################ BİDE BOYU TAHMİN EDELİM ###########

boy=s2.iloc[:,3:4].values
print(boy)
#########dikkat values kullanma 
solkolonlar=s2.iloc[:,0:3]
print(solkolonlar)
sağkolonlar=s2.iloc[:,4:7]
print(solkolonlar)

verikolonlar=pd.concat([solkolonlar,sağkolonlar],axis=1)
print(verikolonlar)
x_test,x_train,y_test,y_train = train_test_split(verikolonlar,boy,test_size=0.33,random_state=0)
r2=LinearRegression()
r2.fit(x_train,y_train)
y_pred=r2.predict(x_test)


########BACKWARD ELİMİNATİON
import statsmodels.api as sm 
X=np.append(arr=np.ones((22,1)).astype(int), values=verikolonlar,axis=1)
X_l=verikolonlar.iloc[:,[0,1,2,3,4,5]].values
X_l=np.array(X_l,dtype=float)
model=sm.OLS(boy,X_l).fit()
print(model.summary())







