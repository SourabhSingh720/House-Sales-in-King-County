# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 17:13:57 2022

@author: forev
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

df=pd.read_csv('kc_house_data.csv')

#print(df.info())
# =============================================================================
# for column in df.columns:
#       print(df[column].value_counts())
#       print('*'*20)
# =============================================================================

df=df.drop(['id','date'],axis=1)
# =============================================================================
# 
# df['renovated(y/n)'] = df['yr_renovated'].apply(lambda x : 1 if x!=0 else 0)
# df.drop('yr_renovated', axis=1, inplace=True)
# 
# =============================================================================
df = df.drop('floors', axis = 1)

X = df.iloc[:, 1:]
y = df.iloc[:, 0]

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

scaler_minmax = MinMaxScaler()
X_train = scaler_minmax.fit_transform(X_train)
X_test = scaler_minmax.transform(X_test)

#Linear Regression

lr=LinearRegression()
lr.fit(X_train,y_train)
pred_lr=lr.predict(X_test)
#print(r2_score(y_test,pred_lr))
rmse_lr = (mean_squared_error(y_test, pred_lr))
rmse_lr = np.sqrt(rmse_lr)


#Lasso

la=Lasso()
la.fit(X_train,y_train)
pred_la=la.predict(X_test)
#print(r2_score(y_test,pred_la))
rmse_la = (mean_squared_error(y_test, pred_la))
rmse_la = np.sqrt(rmse_la)

#Ridge

Rd=Ridge()
Rd.fit(X_train,y_train)
pred_Rd=Rd.predict(X_test)
#print(r2_score(y_test,pred_Rd))
rmse_Rd = (mean_squared_error(y_test, pred_Rd))
rmse_Rd = np.sqrt(rmse_Rd)

#Random Forest Regression

rf_reg = RandomForestRegressor()
rf_reg.fit(X_train, y_train)
pred_rf = rf_reg.predict(X_test)
#print(r2_score(y_test,pred_rf))
rmse_rf = (mean_squared_error(y_test, pred_rf))
rmse_rf = np.sqrt(rmse_rf)

#Polynomial Regression

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

plr=LinearRegression()
plr.fit(X_train_poly,y_train)
pred_plr_poly=plr.predict(X_test_poly)
#print(r2_score(y_test,pred_lr_poly))
rmse_plr = (mean_squared_error(y_test, pred_plr_poly))
rmse_poly = np.sqrt(rmse_plr)


results = pd.DataFrame(columns =['RMSE', 'R2_score'], 
                       index = ['Linear Regression', 'Lasso Regression', 'Random Forest Regression', 'Polynomial Regression'])

results.loc['Linear Regression'] = [round(rmse_lr, 2), r2_score(y_test, pred_lr)]
results.loc['Lasso Regression'] = [round(rmse_la, 2), r2_score(y_test, pred_la)]
results.loc['Random Forest Regression'] = [round(rmse_rf, 2), r2_score(y_test, pred_rf)]
results.loc['Polynomial Regression'] = [round(rmse_poly, 2), r2_score(y_test, pred_plr_poly)]

print(results)


