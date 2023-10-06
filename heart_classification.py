# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 17:18:55 2023

@author: SHYMAA
"""
import pandas as pd
import numpy as np
#from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import  matplotlib.pyplot as plt
data=pd.read_csv('D:\\MachineCourse\\MachineLearnig\\Data\\Data\\2.5 SVC\\heart.csv')
print(data.head())
print(data.shape)
#================= clean data =================#
data=data.dropna()
#========================= positive data =======================#
positive=data[data['target'].isin([1])]
print("====================== The People have heart =====================")
print(positive)
negative=data[data['target'].isin([0])]
print("====================== The People have not heart =====================")
print(negative)
# =============================================================================
# fig,ax=plt.subplots(figsize=(8,5))
# ax.scatter(positive['sex'],positive['cp'],
#           label='Have Heart',s=50,c='r',
#            
#            )
# 
# ax.scatter(negative['sex'],negative['cp'],
#           label='Have no Heart',marker='X',c='b',s=50
#            
#            )
# #data=data.drop_duplicates()
# =============================================================================
print(data.value_counts())
print(data.info())
print(data.describe())
print(data.isnull())
#=======================================================#
sc=StandardScaler()
x=data.iloc[:,:-1]
y=data.iloc[:,-1:]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=44)
print(x_train)
print(y_train)
print(x_test)
print(y_test)
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

#modelSVC=SVC(kernel='poly',degree=2,)
#modelSVC=LogisticRegression()
model= GaussianNB()
#modelSVC=GradientBoostingClassifier(random_state=(0))
model.fit(x_train,y_train)
y_predict=model.predict(x_test)
print(y_predict)
print("accuracy = ",accuracy_score(y_test, y_predict)*100) #88%
cm=confusion_matrix(y_test, y_predict)
print(cm)
sns.heatmap(cm)
plt.show()

x_test=np.matrix([57,1,0,110,201,0,1,126,1,1.5,1,0,1])
print(model.predict(x_test))


#==========================================================================#







