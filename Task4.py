import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("C:\IRIS.csv")
print(df)
print(df.head())
print(df.isnull().sum())

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(df.petal_length,df.petal_width)
ax.set_xlabel('PetalLengthCm')
ax.set_ylabel('PetalWidthCm')
ax.set_zlabel('Species')
plt.title('3D Scatter Plot Example')
plt.show()
print(df.describe())
sns.pairplot(df)
plt.show()
fig1=plt.figure()
ax=fig1.add_subplot(111,projection='3d')
ax.scatter(df.sepal_length,df.sepal_width)
ax.set_xlabel('SepalLengthCm')
ax.set_ylabel('SepalWidthCm')
ax.set_zlabel('Species')
plt.title('3D Scatter Plot Example')
plt.show()
sns.scatterplot(data=df,x="sepal_length",y="sepal_width")
plt.show()
sns.scatterplot(data=df,x="petal_length",y="petal_width",hue="species")
plt.show()
k_rng=range(1,10)
ss=[]
for k in k_rng:
    kn=KMeans(n_clusters=k)
    kn.fit(df[['petal_length','petal_width']])
    ss.append(kn.inertia_)
plt.xlabel('k_rng')
plt.ylabel("SUm of squared errors")
plt.plot(k_rng,ss)
plt.show()
kn=KMeans(n_clusters=3,random_state=0)
y_predict=kn.fit_predict(df[['petal_length','petal_width']])
print(y_predict)
df['cluster']=y_predict
print(df.head(150))
data=df.values
x=data[:,0:4]
y=data[:,4]
print(x)
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_test)
#vector algorithm

model_svc=SVC()
model_svc.fit(x_train,y_train)
predict1=model_svc.predict(x_test)
print(accuracy_score(y_test,predict1))
#Logistic Regression

model_lr=LogisticRegression()
model_lr.fit(x_train,y_train)

# Accuracy Prediction

predict2=model_lr.predict(x_test)
print(accuracy_score(y_test,predict2)*100)
for i in range(len(predict1)):
    print(y_test[i],predict1[i])

#Decision Tree Classifier

model_dt=DecisionTreeClassifier()
model_dt.fit(x_train,y_train)

#Accuracy Prediction

predict3=model_svc.predict(x_test)
print(accuracy_score(y_test,predict3))

print(classification_report(y_test,predict2))
x_n=np.array([[3,2,1,0.2],[4.9,2.2,3.8,1.1],[5.3,2.5,4.6,1.9]])
predict=model_svc(x_n)
print("Prediction of species",format(predict))