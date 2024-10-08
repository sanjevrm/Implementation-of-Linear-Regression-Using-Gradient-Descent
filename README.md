
# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
```
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by:JAGADEESH J
RegisterNumber:212223110015

from google.colab import drive
drive.mount('/content/gdrive')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1,y,learning_rate=0.01,num_iters=1000):
  x=np.c_[np.ones(len(x1)),x1]
  theta=np.zeros(x.shape[1]).reshape(-1,1)
  for c in range(num_iters):
    predictions=(x).dot(theta).reshape(-1,1)
    errors=(predictions-y).reshape(-1,1)
    theta-=learning_rate*(1/len(x1))*x.T.dot(errors)
    return theta 
*/
```
## Dataset:
```
a=pd.read_csv('/content/gdrive/MyDrive/50_Startups.csv')
a
```
## Output:
![image](https://github.com/user-attachments/assets/36c44145-16a6-4fa1-b74d-40d78f5b3064)
![image](https://github.com/user-attachments/assets/2246898d-153f-4fc2-8b30-e228694bb9e4)



## Head and Tail:
```
print(a.head())
print(a.tail())
```
# Output:
![image](https://github.com/user-attachments/assets/ad888cde-8c02-4cd2-97f3-d9c0e401d69d)

## Information of Dataset:
```
a.info()
```
## Output:
![image](https://github.com/user-attachments/assets/7520c762-2bc8-4c37-8aff-b9f6ea590b57)

## x and y value:
```
x=a.iloc[:,:-1].values
print(x)
y=a.iloc[:,-1].values
print(y)
```
## Output:
![image](https://github.com/user-attachments/assets/449058e2-69c7-4a4b-817b-96f95251500b)

## StandardScaler:
```
scaler=StandardScaler()
x=scaler.fit_transform(x)
print(x)
y=y.reshape(-1,1)
y=scaler.fit_transform(y)
print(y)

```
## Output:
![image](https://github.com/user-attachments/assets/2c4c43bc-da9d-44a0-93ac-f8439aea5850)

## Final pridiction:
```
x1=x.astype(float)
x1_scaled=scaler.fit_transform(x1)
y1_scaled=scaler.fit_transform(y)
theta=linear_regression(x1_scaled,y1_scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value:{pre}")
```
## Output:
![image](https://github.com/user-attachments/assets/96d08bd6-7f9f-4dcb-8376-3a9f417ad6b3)
## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
