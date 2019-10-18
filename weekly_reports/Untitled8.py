#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

#load packages
import sys #sistem parametrelerine erişim için kullanılır https://docs.python.org/3/library/sys.html
print("Python version: {}". format(sys.version))

import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features
print("pandas version: {}". format(pd.__version__))

import matplotlib #collection of functions for scientific and publication-ready visualization
print("matplotlib version: {}". format(matplotlib.__version__))

import numpy as np #bilimsel hesaplar için hazırlanmış topluluk kütüphanesi
print("NumPy version: {}". format(np.__version__))

import scipy as sp #ileri matematik ve bilim için hazırlanmış bir kütüphane
print("SciPy version: {}". format(sp.__version__)) 

import IPython
from IPython import display #Jupyter notebook'da daha iyi bir görüntü alınabilmesi için kullanılır.
print("IPython version: {}". format(IPython.__version__)) 

import sklearn #çözümlerde kullanacağımız fonksiyonların bulunduğu kütüphane
print("scikit-learn version: {}". format(sklearn.__version__))

#misc libraries
import random
import time


#uyarıları yoksaymak için
import warnings
warnings.filterwarnings('ignore')
print('-'*25)



# Dizindeki input datalarını gösterebilmek için kullanılır.


from subprocess import check_output
print(check_output(["ls", "/home/administrator/PycharmProjects/Uni/weekly_reports/data/titanic"]).decode("utf8"))


# In[2]:


#Yaygın Model Algoritmaları
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Yaygın Model Yardımcıları
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Görselleştirme
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.plotting import scatter_matrix

#Görselleştirme varsayılan ayarlarının düzenlenmesi
#%matplotlib inline = Jupyter Notebook'da plotların gözükmesi için
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8


# In[3]:


#training datası dizinden okunur: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
data_raw = pd.read_csv('/home/administrator/PycharmProjects/Uni/weekly_reports/data/titanic/train.csv')



#test datası dizinden okunur
data_val  = pd.read_csv('/home/administrator/PycharmProjects/Uni/weekly_reports/data/titanic/test.csv')


#data ile oynamak için bir kopya oluşturulur.
#remember python assignment or equal passes by reference vs values, so we use the copy function: https://stackoverflow.com/questions/46327494/python-pandas-dataframe-copydeep-false-vs-copydeep-true-vs
data1 = data_raw.copy(deep = True)

#datasetlerin ikisini de aynı anda temizleyebilmek için
data_cleaner = [data1, data_val]


#datayı önizleme
print (data_raw.info()) #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.info.html
#data_raw.head() #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.head.html
#data_raw.tail() #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.tail.html
data_raw.sample(10) #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sample.html


# In[4]:


print('Train columns with null values:\n', data1.isnull().sum())
print("-"*10)

print('Test/Validation columns with null values:\n', data_val.isnull().sum())
print("-"*10)

data_raw.describe(include = 'all')


# In[5]:


for dataset in data_cleaner:    
    #complete missing age with median
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)

    #complete embarked with mode
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)

    #complete missing fare with median
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
    
#delete the cabin feature/column and others previously stated to exclude in train dataset
drop_column = ['PassengerId','Cabin', 'Ticket']
data1.drop(drop_column, axis=1, inplace = True)

print('Train columns with null values:\n' , data1.isnull().sum())
print("-"*10)
print('Test/Validation columns with null values:\n', data_val.isnull().sum())


# In[ ]:




