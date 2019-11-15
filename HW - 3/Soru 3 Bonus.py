#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import resample


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


rf = RandomForestClassifier(n_estimators=50,max_features=50,bootstrap= False)
rf.fit(xtr,ytr)


# In[12]:


print("Eğitme Kümesi için doğruluk : " , round(rf.score(xtr,ytr),8))


# In[13]:


print("Test Kümesi için doğruluk : " , round(rf.score(xtst,ytst),8))


# In[38]:


from sklearn.utils import resample
boot = resample(df, replace=True, n_samples=50, random_state=1)


# In[39]:


print('Bootstrap Sample:', boot)


# In[36]:


# out of bag observations
oob = [x for x in df if x not in boot]


# In[37]:


print('OOB Sample' , oob)


# In[ ]:




