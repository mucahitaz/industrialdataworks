#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# In[2]:


df = pd.read_csv('/home/administrator/PycharmProjects/Uni/HW - 3/mnist_23/mnist_train23.csv')


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.shape


# In[6]:


test = pd.read_csv('/home/administrator/PycharmProjects/Uni/HW - 3/mnist_23/mnist_test23.csv')


# In[7]:


test.head()


# In[8]:


test.shape


# In[9]:


test.describe()


# In[10]:


xtr = df.iloc[:,1:]
ytr = df.iloc[:,0]

xtst = test.iloc[:,1:]
ytst = test.iloc[:,0]


# In[11]:


bg = BaggingClassifier(DecisionTreeClassifier(),max_samples=50, 
                       max_features=1.0, n_estimators=25, 
                       bootstrap = True)
bg.fit(xtr,ytr)


# In[14]:


print("Eğitme Kümesi için doğruluk : " , round(bg.score(xtr,ytr),4))


# In[16]:


print("Test Kümesi için doğruluk : " , round(bg.score(xtst,ytst),4))


# In[ ]:




