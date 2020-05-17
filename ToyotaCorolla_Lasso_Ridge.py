#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt


# In[30]:


ToyotaCorolla = pd.read_csv(r"C:\Users\srira\Desktop\Ram\Data science\Course - Assignments\Module 8 - Lasso-Ridge Regression\Datasets\ToyotaCorolla.csv",encoding='unicode_escape')
ToyotaCorolla.head()


# In[31]:


ToyotaCorolla.drop(['Model','Mfg_Year','Id','Color'],axis=1,inplace=True)


# In[32]:


ToyotaCorolla['Fuel_Type'].value_counts()


# In[33]:


#Setting Dummy variables
ToyotaCorolla = pd.get_dummies(data=ToyotaCorolla,columns=['Fuel_Type'],drop_first=True)


# In[34]:


ToyotaCorolla.head()


# In[22]:


#RMSE of perfect multiple linear model with all the aspects
# RMSE: 1231.6935032422946


# In[23]:


#Buidling model with Lasso and Ridge


# In[24]:


#Lasso model


# In[35]:


lasso = Lasso(alpha=0.8,normalize=True)


# In[36]:


lasso.fit(ToyotaCorolla.drop(['Price'],axis=1),ToyotaCorolla['Price'])


# In[37]:


pred = lasso.predict(ToyotaCorolla.drop(['Price'],axis=1))
resid = ToyotaCorolla['Price'] - pred
rmse = np.sqrt(np.mean(np.square(resid)))
rmse


# In[45]:


plt.figure(figsize=(15,5))
plt.bar(height=lasso.coef_,x=ToyotaCorolla.drop(['Price'],axis=1).columns);plt.axhline(y=0,color='r')


# In[38]:


#Ridge model


# In[39]:


ridge = Ridge(alpha=0.8,normalize=True)


# In[40]:


ridge.fit(ToyotaCorolla.drop(['Price'],axis=1),ToyotaCorolla['Price'])


# In[41]:


pred = ridge.predict(ToyotaCorolla.drop(['Price'],axis=1))
resid = ToyotaCorolla['Price'] - pred
rmse = np.sqrt(np.mean(np.square(resid)))
rmse


# In[46]:


plt.figure(figsize=(15,5))
plt.bar(height=ridge.coef_,x=ToyotaCorolla.drop(['Price'],axis=1).columns);plt.axhline(y=0,color='r')

