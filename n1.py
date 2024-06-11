import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
x = pd.read_csv("C:\Titanic-Dataset.csv")
print(x)
x.head(10)
print(x)
x.describe()
print(x)
print(x.shape)
print(x.info)
print(x.isnull().sum())   #calculate null values
x=x.drop(columns='Cabin',axis=1)  #drop cabin table
x['Age'].fillna(x['Age'].mean(),inplace=True)  #Replace Missing values of Age with mean of Age
print(x.isnull().sum())
print(x['Embarked'].mode())  #mode of the Embarked
print("Modes here")
print(x['Embarked'].mode()[0])
x['Embarked'].fillna(x['Embarked'].mode()[0],inplace=True) #Replace the missing value in the Embarked
print(x.isnull().sum())
#getting statistical measures of data
#find no of people survived or not survived
print(x["Survived"].value_counts())
sns.set()
#make count plot for "survived" colm
sns.countplot(x='Survived',data=x)
plt.show()
sns.countplot(x='Sex',data=x)
plt.show()
print(x['Sex'].value_counts())
#no of survivors gender wise
sns.countplot(x='Sex',hue='Survived',data=x)
plt.show()
sns.countplot(x='Pclass',hue='Survived',data=x)
plt.show()
sns.countplot(x='Pclass',data=x)
plt.show()
sns.countplot(x='Embarked',data=x)
plt.show()
#Encoding the Categorical column
print(x['Sex'].value_counts())
print(x['Embarked'].value_counts())
#convert categorial colm
x.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':2}},inplace=True)
print(x.head())
print(x)
x1=x.drop(columns=['PassengerId','Name','Ticket','Survived'],axis=1)
y1=x['Survived']
print(x1)
print(y1)
#split data into training and test data


x_t1,x_t2,y_t1,y_t2=train_test_split(x1,y1,test_size=0.2,random_state=2)
print(x.shape,x_t1.shape,x_t2.shape)
model=LogisticRegression()
#Training logistic regression model
model.fit(x_t1,y_t1)

#model evaluation
#accuracy of training data
x_t_pred=model.predict(x_t1)
print(x_t_pred)
#training data accuracy
train_data_acc=accuracy_score(y_t1,x_t_pred)
print("Accuracy score of training data = ",train_data_acc)

#accuracy of test data
x_test_pred=model.predict(x_t2)
print(x_test_pred)

#training data accuracy
test_data_acc=accuracy_score(y_t2,x_test_pred)
print("Accuracy score of test data = ",test_data_acc)

