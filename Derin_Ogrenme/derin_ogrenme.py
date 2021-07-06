# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 22:32:49 2021

@author: ahmet
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

veriler = pd.read_csv("model.csv")


X=veriler.iloc[:,3:13].values
Y=veriler.iloc[:,13].values

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
X[:,1]=le.fit_transform(X[:,1])

le2=preprocessing.LabelEncoder()
X[:,2]=le.fit_transform(X[:,2])


from sklearn.preprocessing import  OneHotEncoder
from sklearn.compose import ColumnTransformer

ohe=ColumnTransformer([("ohe",OneHotEncoder(dtype=float),[1])],remainder="passthrough")

X=ohe.fit_transform(X)
X=X[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y ,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)



#####DERİN ÖĞRENME ADIMLARI 
import keras
from keras.models import Sequential 
from keras.layers import Dense

derin=Sequential()
derin.add(Dense(6 ,init="uniform",activation="relu",input_dim=11))
derin.add(Dense(6 ,init="uniform",activation="relu"))
derin.add(Dense(1 ,init="uniform",activation="sigmoid"))
derin.compile(optimizer="adam", loss="binary_crossentropy",metrics=["accuracy"])
derin.fit(X_train,y_train,epochs=100)
y_pred=derin.predict(X_test)