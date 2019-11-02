# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# load packages
import sys  # access to system parameters https://docs.python.org/3/library/sys.html

print("Python version: {}".format(sys.version))

import \
    pandas as pd  # collection of functions for data processing and analysis modeled after R dataframes with SQL like features

print("pandas version: {}".format(pd.__version__))

import matplotlib  # collection of functions for scientific and publication-ready visualization

print("matplotlib version: {}".format(matplotlib.__version__))

import numpy as np  # foundational package for scientific computing

print("NumPy version: {}".format(np.__version__))

import scipy as sp  # collection of functions for scientific computing and advance mathematics

print("SciPy version: {}".format(sp.__version__))

# import IPython
# from IPython import display #pretty printing of dataframes in Jupyter notebook
# print("IPython version: {}". format(IPython.__version__))

import sklearn  # collection of machine learning algorithms

print("scikit-learn version: {}".format(sklearn.__version__))

# misc libraries
import random
import time

# ignore warnings
import warnings

warnings.filterwarnings('ignore')
print('-' * 25)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "/home/administrator/PycharmProjects/Uni/weekly_reports/data/titanic"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

# Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.plotting import scatter_matrix

# Configure Visualization Defaults
# %matplotlib inline = show plots in Jupyter Notebook browser
# get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12, 8

# import data from file: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
data_raw = pd.read_csv('/home/administrator/PycharmProjects/Uni/weekly_reports/data/titanic/train.csv')

# a dataset should be broken into 3 splits: train, test, and (final) validation
# the test file provided is the validation file for competition submission
# we will split the train set into train and test data in future sections
data_val = pd.read_csv('/home/administrator/PycharmProjects/Uni/weekly_reports/data/titanic/test.csv')

# to play with our data we'll create a copy
# remember python assignment or equal passes by reference vs values, so we use the copy function: https://stackoverflow.com/questions/46327494/python-pandas-dataframe-copydeep-false-vs-copydeep-true-vs
data1 = data_raw.copy(deep=True)

# however passing by reference is convenient, because we can clean both datasets at once
data_cleaner = [data1, data_val]

# preview data
print(data_raw.info())  # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.info.html
print(data_raw.head())  # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.head.html
# data_raw.tail() #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.tail.html
data_raw.sample(10)  # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sample.html

print('Train columns with null values:\n', data1.isnull().sum())
print("-" * 10)

print('Test/Validation columns with null values:\n', data_val.isnull().sum())
print("-" * 10)

data_raw.describe(include='all')

for dataset in data_cleaner:
    # complete missing age with median
    dataset['Age'].fillna(dataset['Age'].median(), inplace=True)

    # complete embarked with mode
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)

    # complete missing fare with median
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)

# delete the cabin feature/column and others previously stated to exclude in train dataset
drop_column = ['PassengerId', 'Cabin', 'Ticket']
data1.drop(drop_column, axis=1, inplace=True)

print(data1.isnull().sum())
print("-" * 10)
print(data_val.isnull().sum())

y = data_raw.Survived.values
print(y)

total = [data_raw, data_val,data1]
for dataset in total:
    dataset.loc[dataset['Age'] <= 18, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

print(data1)
print(data1.head())

titanic=pd.get_dummies(titanic,columns=['Sex','Embarked'],drop_first=True)
titanic_test=pd.get_dummies(titanic_test,columns=['Sex','Embarked'],drop_first=True)