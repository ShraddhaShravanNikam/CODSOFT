import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score
import seaborn as sns
df = pd.read_csv("C:\Titanic-Dataset1.csv")
print(df)
print(df.head)
print(df.shape)
sns.pairplot(df,x_vars=['TV','Radio','Newspaper'],y_vars='Sales',kind='scatter')
plt.show()
df['TV'].plot.hist(bins=10,xlabel="TV")
plt.show()
df['Radio'].plot.hist(bins=10,xlabel="Radio")
plt.show()
df['Newspaper'].plot.hist(bins=10,xlabel="Newspaper")
plt.show()
sns.heatmap(df.corr(),annot=True)
plt.show()
x_trn,x_tst,y_trn,y_tst=train_test_split(df[['TV']],df[['Sales']],test_size=0.3,random_state=0)
print(x_trn)
print(y_trn)
print(x_tst)
print(y_tst)
model=LinearRegression()
model.fit(x_trn,y_trn)
res=model.predict(x_tst)
print("X_tst",res)

print("====",model.coef_)
print(model.intercept_)
plt.plot(res)
plt.show()
plt.scatter(x_tst,y_tst)
plt.plot(x_tst,7.14382225+0.05473199* x_tst,'r')
plt.show()