# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 15:23:09 2021

@author: ahmet
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler=pd.read_csv("monthsandsales.csv")
print(veriler)

aylar=veriler["Aylar"]
print(aylar)
satislar=veriler["Satislar"]
print(satislar)

from sklearn.model_selection import train_test_split      ###bağımlı satış 
x_train, x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)

#verilerin olceklenmesi 

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train=sc.fit_transform(y_train)
Y_test=sc.fit_transform(y_test)

####DOĞRUSAL MODEL İNSAŞI 
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)

tahmin = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))

plt.title("aylara göre satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")
