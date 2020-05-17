#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt


# In[7]:


Computer_Data = pd.read_csv(r"C:\Users\srira\Desktop\Ram\Data science\Course - Assignments\Module 8 - Lasso-Ridge Regression\Datasets\Computer_Data.csv")
Computer_Data.head(10)


# In[8]:


Computer_Data.drop(['Unnamed: 0'],axis=1,inplace=True)
Computer_Data.head(10)


# In[18]:


#Dummy variables for categorical data
Computer_Data = pd.get_dummies(Computer_Data,columns=['cd','multi','premium'],drop_first=True)


# In[19]:


Computer_Data.head()


# In[10]:


#Pulling results from multi linear regression


# In[11]:


# rmse = 258.7482165316277


# In[12]:


#Let's build Lasso & Ridge models to check accuracy


# In[20]:


indep_vars = Computer_Data.drop(['price'],axis=1)


# In[21]:


lasso = Lasso(alpha=0.8,normalize=True)


# In[22]:


lasso.fit(indep_vars,Computer_Data['price'])


# In[23]:


pred = lasso.predict(indep_vars)
resid = Computer_Data['price'] - pred
rmse = np.sqrt(np.mean(np.square(resid)))
rmse


# In[29]:


plt.bar(height=lasso.coef_,x=indep_vars.columns);plt.axhline(y=0,color='r')


# In[36]:


lasso.coef_


# In[30]:


#Building Ridge model


# In[31]:


ridge = Ridge(alpha=0.8,normalize=True)


# In[32]:


ridge.fit(indep_vars,Computer_Data['price'])


# In[33]:


pred = ridge.predict(indep_vars)
resid = Computer_Data['price'] - pred
rmse = np.sqrt(np.mean(np.square(resid)))
rmse


# In[34]:


plt.bar(height=ridge.coef_,x=indep_vars.columns);plt.axhline(y=0,color='r')


# In[35]:


ridge.coef_


# In[ ]:


# Conclusion: Tranformed multiple model has better RMSE than Lasso and Ridge

