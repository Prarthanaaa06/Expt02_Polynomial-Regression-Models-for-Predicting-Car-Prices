# BLENDED_LEARNING
# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement Linear and Polynomial Regression models for predicting car prices.
Developed by: Prarthana D
RegisterNumber:  21225230213
*/

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#these below 2 lines is where the diff likes b/w ex1 and ex2
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import matplotlib.pyplot as plt
df = pd.read_csv('encoded_car_data (1).csv')
print(df.head())
X = df[['enginesize','horsepower','citympg','highwaympg']]
y = df['price']
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=42)
#1 Linear Regression (wt scaling)
lr = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
lr.fit(X_train, y_train)
y_pred_linear = lr.predict(X_test)
#2 Polynomial Regression (degree = 2)
poly_model = Pipeline([
    ('poly',PolynomialFeatures(degree=2)),
    ('scaler', StandardScaler()),
    ('model',LinearRegression())
])
poly_model.fit(X_train,y_train)
y_pred_poly = poly_model.predict (X_test)
#Evaluate models
print("Name: PRARTHANA D")
print("Reg. No: 212225230213")
print("\nLINEAR REGRESSION:")

mse = mean_squared_error(y_test,y_pred_linear)
print("MSE: ",mse)
print(f"MAE : {mean_absolute_error(y_test,y_pred_linear)}")
r2score = r2_score(y_test,y_pred_linear)
print("R² Score: ", r2score)

print("\nPOLYNOMIAL REGRESSION:")
print(f"MSE : {mean_squared_error(y_test,y_pred_poly):.2f}")
print(f"MAE : {mean_absolute_error(y_test,y_pred_poly):.2f}")
print(f"R² Score: {r2_score(y_test,y_pred_poly):.2f}")

#Plot acatual VS Predicted
plt.figure(figsize=(10,5))
plt.scatter(y_test,y_pred_linear,label='Linear',alpha=0.6)
plt.scatter(y_test,y_pred_poly,label ="Polynomial (degree = 2)",alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()], 'r--', label='Perfect Predication')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear VS Polynomail Regression")
plt.legend()
plt.show()

```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
<img width="1127" height="563" alt="image" src="https://github.com/user-attachments/assets/4ebb94bf-529b-46cd-b8c1-9edcc26f1343" />
<img width="512" height="308" alt="image" src="https://github.com/user-attachments/assets/49f1c719-e2f3-4455-9d76-05f75d92be4c" />
<img width="1428" height="641" alt="image" src="https://github.com/user-attachments/assets/4e309025-175b-44ba-a124-aa1e16497a32" />


## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
