#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import pandas as pd
import numpy as np


# In[2]:


model=pickle.load(open('catmode','rb'))


# In[3]:


model


# In[4]:


data_test=pd.read_csv('test.csv')


# In[5]:


x=data_test.drop('id',axis=1)


# In[6]:


data_test['loss']=np.exp(model.predict(x))


# In[7]:


data_test


# In[8]:


data_test.to_csv('result.csv')


# In[ ]:




