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


# bg = BaggingClassifier(DecisionTreeClassifier(), 
#                        max_features=1.0, n_estimators=25, 
#                        bootstrap = True)


# In[24]:


def BaggingFunc(samples):
    bg = BaggingClassifier(DecisionTreeClassifier(),max_samples=samples, 
                       max_features=1.0, n_estimators=25, 
                       bootstrap = True)
    bg.fit(xtr,ytr) 
    score = round(bg.score(xtst,ytst),4)
    print("{} örnekli test Kümesi için doğruluk : ".format(samples) , round(bg.score(xtst,ytst),4))
    return score


# In[25]:


BaggingFunc(50)


# In[26]:


BaggingFunc(20)


# In[27]:


BaggingFunc(5)


# In[28]:


BaggingFunc(10)


# In[38]:


names = ['05 Samples' , '10 Samples', '20 Samples', '50 Samples']
accuracy = [BaggingFunc(5),BaggingFunc(10),BaggingFunc(20),BaggingFunc(50)]


# In[39]:


import seaborn as sns
ax = sns.barplot(x= names, y= accuracy)


# In[40]:


import seaborn as sns
sns.set(style="darkgrid")


# Plot the responses for different events and regions
sns.lineplot(x= names, y= accuracy)


# In[ ]:




