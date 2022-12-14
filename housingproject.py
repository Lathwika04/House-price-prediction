import pandas as pd
import math

dataset=pd.read_csv("HousingData.csv")
print(dataset.shape) #506 14

for i in range(0,14):
    mn=dataset.iloc[:,i].values.mean()
    dataset.fillna(mn,inplace=True)

#print(dataset.to_string())


import sklearn.linear_model
from sklearn.linear_model import LinearRegression
import seaborn as sb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


x=dataset.iloc[:,0:13].values
y=dataset.iloc[:,13:14].values
l=len(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

acc1=r2_score(y_train,model.predict(x_train))*100
acc=r2_score(y_test,y_pred)*100
print(acc1)
print(acc)