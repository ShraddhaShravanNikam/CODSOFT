import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import seaborn as sns

cd = pd.read_csv('C:\creditcard.csv')
print(cd)
#First five rows
print(cd.head())
#Last five rows
print(cd.tail())
#Dataset Information
print(cd.info())
#missing values
print(cd.isnull().sum())
#check fraud transaction
print(cd['Class'].value_counts())
#analysis the data
nof=cd[cd.Class==0]
fraud=cd[cd.Class==1]
print("Not Fraud",nof)
print("Fraud=",fraud)
print(nof.shape)
print(fraud.shape)
print(nof.Amount.describe())
print(fraud.Amount.describe())
#compare the values for both transaction
print(cd.groupby('Class').mean())
legit=nof.sample(n=492)
new=pd.concat([legit,fraud],axis=0)
print(new)
print(new.head())
print(new['Class'].value_counts())
print(new.groupby('Class').mean())
x=new.drop(columns='Class',axis=1)
y=new['Class']
print("X=",x)
print("Y=",y)
#Split the data into training data and testing data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)
print(x.shape,x_train.shape,x_test.shape)
model=LogisticRegression()
model.fit(x_train,y_train)
#accuracy of training data
x_train_pred=model.predict(x_train)
x_train_acc=accuracy_score(x_train_pred,y_train)
print("Accuracy Training data",x_train_acc)
x_test_pred=model.predict(x_test)
x_test_acc=accuracy_score(x_test_pred,y_test)
print("Accuracy Testing data",x_test_acc)
test_data_acc=accuracy_score(x_test_pred,y_test)
print("Accuracy score of test data accuracy",test_data_acc)
sns.countplot(x='Class',data=cd)
plt.show()




