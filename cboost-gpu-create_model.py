#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


pip install catboost


# In[3]:


import catboost


# In[4]:


data_train=pd.read_csv('train.csv')


# In[5]:


data_train.head()


# In[6]:


data_train.isnull().sum().sum()


# In[7]:


data_test=pd.read_csv('test.csv')


# In[8]:


data_test.head()


# In[11]:


data_train


# In[12]:


import re


# In[13]:


cat_pat=re.compile("^cat([1-9]|[1-9][0-9]|[1-9][0-9][0-9])$")
cont_pat=re.compile("^cont([1-9]|[1-9][0-9]|[1-9][0-9][0-9])$")


# In[14]:


cat_ind=[i for i in range(0,len(data_train.columns)) if cat_pat.match(data_train.columns[i])]


# In[15]:


cont_ind=[i for i in range(0,len(data_train.columns)) if cont_pat.match(data_train.columns[i])]


# In[18]:


plt.figure(figsize=(20,12))
sns.distplot(data_train['loss'])


# In[19]:


plt.figure(figsize=(20,12))
sns.distplot(np.log(data_train['loss']))


# In[20]:


from catboost import CatBoostRegressor
model=CatBoostRegressor(iterations=200,depth=6,learning_rate=0.05,eval_metric='MAE',verbose=10,task_type='GPU',save_snapshot=True,snapshot_file='shima',snapshot_interval=10)


# In[21]:


x=data_train.drop(['id','loss'],axis=1)
y=np.log(data_train['loss'])


# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=13)


# In[24]:


model.fit(x_train,y_train,np.asarray(cat_ind)-1,eval_set=(x_test,y_test))


# In[26]:


del x_train
del x_test
del y_train
del y_test


# In[27]:


del data_train


# In[28]:


import pickle


# In[29]:


with open ('catmodegpu','wb')as f:
    pickle.dump(model,f)


# In[ ]:




