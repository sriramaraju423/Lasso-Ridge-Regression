#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import statsmodels.formula.api as smf
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge


# In[38]:


Startups = pd.read_csv(r"C:\Users\srira\Desktop\Ram\Data science\Course - Assignments\Module 8 - Lasso-Ridge Regression\Datasets\50_Startups.csv")
Startups.head(10)


# In[39]:


Startups = Startups.rename(columns={'R&D Spend':'RandD_Spend','Marketing Spend':'Marketing_Spend'})


# In[40]:


Startups.head(10)


# In[3]:


#EDA


# In[41]:


Startups.shape


# In[42]:


Startups.describe()


# In[8]:


#Outlier handling


# In[43]:


Startups.plot(kind='box')


# In[21]:


# Boxplot seems good, outliers can be ignored


# In[48]:


# Looking into pair plot just to check correlation b/w all the variables


# In[49]:


sb.pairplot(Startups)


# In[ ]:


# WE can see multi colinearity problem b/w RandD and Marketing


# In[22]:


#Multiple linear regression model building


# In[45]:


multiple_model = smf.ols('Profit ~ RandD_Spend+Administration+Marketing_Spend', data=Startups).fit()


# In[47]:


multiple_model.summary()


# In[ ]:


#Good adjusted R2 and variable 'Administration' is not significant


# In[53]:


independent_var=Startups.drop(['Profit','State'],axis=1)


# In[54]:


pred = multiple_model.predict(independent_var)


# In[55]:


resid = Startups['Profit'] - pred


# In[56]:


rmse = np.sqrt(np.mean(np.square(resid)))
rmse


# In[50]:


#Let's build lasso and ridge model


# In[ ]:


#Building Lasso model


# In[77]:


lasso = Lasso(alpha=0.5,normalize=True)


# In[78]:


lasso.fit(independent_var,Startups['Profit'])


# In[79]:


lasso.coef_


# In[80]:


pred = lasso.predict(independent_var)


# In[81]:


resid = Startups['Profit'] - pred


# In[82]:


rmse = np.sqrt(np.mean(np.square(resid)))
rmse


# In[83]:


plt.bar(height=pd.Series(lasso.coef_),x=pd.Series(independent_var.columns))


# In[84]:


#There isn't much change with Lasso let's try ridge


# In[88]:


ridge = Ridge(alpha=0.9,normalize=True)


# In[89]:


ridge.fit(independent_var,Startups['Profit'])


# In[90]:


pred = ridge.predict(independent_var)
resid = Startups['Profit'] - pred
rmse = np.sqrt(np.mean(np.square(resid)))
rmse


# In[91]:


plt.bar(height=pd.Series(ridge.coef_),x=pd.Series(independent_var.columns))


# In[ ]:


# Conclusion: Multiple OLS model is better than lass and ridge


# In[75]:


independent_var.columns


# In[46]:


Startups.columns

