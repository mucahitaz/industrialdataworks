#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.model_selection import cross_val_score, KFold,StratifiedKFold,train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression


from subprocess import check_output
print(check_output(["ls","../data/titanic"]).decode("utf8"))

train_data = pd.read_csv('../data/titanic/train.csv', dtype={'Age': np.float16})

train_data.head()


# In[2]:


print('Train columns with null values:\n', train_data.isnull().sum())
print("-" * 10)


# In[3]:


data = [train_data]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)
train_data['not_alone'].value_counts()


# In[4]:


train_data.head()


# In[5]:


train_data=train_data.drop(['Cabin','PassengerId',"Ticket","Fare"], axis=1)


# In[6]:


train_data.head()


# In[7]:


data = [train_data]
for dataset in data:
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0],inplace=True)
    Embarked = np.zeros(len(dataset))
    Embarked[dataset['Embarked']== 'C'] = 1
    Embarked[dataset['Embarked']== 'Q'] = 2
    Embarked[dataset['Embarked']== 'S'] = 3
    dataset['Embarked'] = Embarked


# In[8]:


train_data['Age'].fillna(train_data['Age'].median(), inplace = True)


# In[9]:


print('Train columns with null values:\n', train_data.isnull().sum())
print("-" * 10)


# In[10]:


train_data.head()


# In[12]:


data = [train_data]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
    # extract titles
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # replace titles with a more common title or as Rare
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # convert titles into numbers
    dataset['Title'] = dataset['Title'].map(titles)
    # filling NaN with 0, to get safe
    dataset['Title'] = dataset['Title'].fillna(0)
train_data = train_data.drop(['Name'], axis=1)


# In[13]:


data = [train_data]
for dataset in data:
    sex = np.zeros(len(dataset))
    sex[dataset['Sex']== 'male'] = 1
    sex[dataset['Sex']== 'female'] = 0
    dataset['Sex'] = sex


# In[14]:


train_data.head()


# In[15]:


X = train_data.drop("Survived", axis=1)
y= train_data["Survived"]

X_train, X_test, Y_train, Y_test = train_test_split(X, y,test_size = 0.20)


# In[16]:


scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)


# In[79]:


"""TEK BİR KARAR AĞACI İLE ÇÖZÜM"""

dtc_scores = []
dtc_scores_test = []
dtc_auc_scores = []

for x in range(1,10):   
    decision_tree = tree.DecisionTreeClassifier(criterion="gini",max_depth=x)
    decision_tree.fit(X_train, Y_train)
    Y_pred=decision_tree.predict(X_test)
    
    acc_dtc = round(decision_tree.score(X_train, Y_train) * 100, 2)
    dtc_auc_score = round(roc_auc_score(Y_test, Y_pred) * 100, 2)
    acc_dtctest = round(decision_tree.score(X_test, Y_test) * 100, 2)
    
    print("Decision Tree Train score for {} depth =".format(x) ,acc_dtc)
    print("Decision Tree Test score for {} depth = ".format(x) ,acc_dtctest)
    print("AUC SCORE for {} depth =".format(x)  ,dtc_auc_score , "\n")
    
    dtc_scores.append(acc_dtc)
    dtc_auc_scores.append(dtc_auc_score)
    dtc_scores_test.append(acc_dtctest)
    

print("Train Scores list:", dtc_scores)
print("Test Scores list:",dtc_scores_test)
print("AUC list:",dtc_auc_scores)


# In[73]:


ax = sns.barplot(x= list(range(1,10)) , y= dtc_scores)


# In[83]:


sns.set(style="darkgrid")
sns.lineplot(x= list(range(1,10)) , y= dtc_scores)


# In[74]:


ax = sns.barplot(x= list(range(1,10)) , y= dtc_scores_test)


# In[84]:


sns.set(style="darkgrid")
sns.lineplot(x= list(range(1,10)) , y= dtc_scores_test)


# In[80]:


ax = sns.barplot(x= list(range(1,10)) , y= dtc_auc_scores)


# In[85]:


sns.set(style="darkgrid")
sns.lineplot(x= list(range(1,10)) , y= dtc_auc_scores)


# In[54]:


"""BAGGING KULLANARAK YAPILAN KARAR AĞACI ÇÖZÜMÜ"""

samples = [5,10,25,50,100,250,500]
scores = []
scores_test = []
auc_scores = []

for x in samples:   
    bg = BaggingClassifier(DecisionTreeClassifier(),max_samples=x, 
                       max_features=1.0, n_estimators=25, 
                       bootstrap = True)
    bg.fit(X_train,Y_train)
    Y_pred = bg.predict(X_test)

    acc_bgdtc = round(bg.score(X_train, Y_train) * 100, 2)
    bg_auc_score = round(roc_auc_score(Y_test, Y_pred) * 100, 2)
    acc_bgdtctest = round(bg.score(X_test, Y_test) * 100, 2)

    print("Bagging Decision Tree Train score for {} samples =".format(x) ,acc_bgdtc)
    print("Bagging Decision Tree Test score for {} samples = ".format(x) ,acc_bgdtctest)
    print("AUC SCORE for {} samples =".format(x)  ,bg_auc_score , "\n")
    scores.append(acc_bgdtc)
    auc_scores.append(bg_auc_score)
    scores_test.append(acc_bgdtctest)
    

print("Train Scores list:", scores)
print("Test Scores list:",scores_test)
print("AUC list:",auc_scores)


# In[42]:


ax = sns.barplot(x= samples , y= scores)


# In[55]:


ax = sns.barplot(x= samples , y= scores_test)


# In[56]:


ax = sns.barplot(x= samples , y= auc_scores)


# In[112]:


"""RANDOM FOREST ÇÖZÜMÜ"""
estimators = [5,10,25,30,40,50]
rfc_scores = []
rfc_scores_test = []
rfc_auc_scores = []

for x in estimators:   
    rf = RandomForestClassifier(criterion='gini',n_estimators=x,bootstrap= False,
                                max_depth=3,max_features='auto')
    rf.fit(X_train,Y_train)
    
    Y_pred = rf.predict(X_test)

    acc_rf = round(rf.score(X_train, Y_train) * 100, 2)
    rf_auc_score = round(roc_auc_score(Y_test, Y_pred) * 100, 2)
    acc_rftest = round(rf.score(X_test, Y_test) * 100, 2)

    print("Random Forest Train score for {} estimators =".format(x) ,acc_rf)
    print("Random Forest Test score for {} estimators = ".format(x) ,acc_rftest)
    print("AUC SCORE for {} estimators =".format(x)  ,rf_auc_score , "\n")
    rfc_scores.append(acc_rf)
    rfc_auc_scores.append(rf_auc_score)
    rfc_scores_test.append(acc_rftest)
    
print("Train Scores list:", rfc_scores)
print("Test Scores list:",rfc_scores_test)
print("AUC list:",rfc_auc_scores)


# In[109]:


sns.set(style="darkgrid")
sns.lineplot(x= estimators , y= rfc_scores)


# In[110]:


ax = sns.lineplot(x= estimators , y= rfc_scores_test)


# In[111]:


ax = sns.lineplot(x= estimators , y= rfc_auc_scores)


# In[137]:


"""ADABOOST"""
estimators = [5,10,25,30,40,50,100,250,500,1000]
adb_rfc_scores = []
adb_rfc_scores_test = []
adb_rfc_auc_scores = []
for x in estimators:
    ad=AdaBoostClassifier(RandomForestClassifier(), n_estimators=x)
    ad.fit(X_train,Y_train)
    Y_pred = ad.predict(X_test)

    acc_adb_rfc = round(ad.score(X_train, Y_train) * 100, 2)
    adb_rfc_auc_score = round(roc_auc_score(Y_test, Y_pred) * 100, 2)
    acc_adb_rfctest = round(ad.score(X_test, Y_test) * 100, 2)

    print("ADABOOST Random Forest Train score for {} estimators =".format(x) ,acc_adb_rfc)
    print("ADABOOST Random Forest Test score for {} estimators = ".format(x) ,acc_adb_rfctest)
    print("AUC SCORE for {} estimators =".format(x)  ,adb_rfc_auc_score , "\n")
    adb_rfc_scores.append(acc_adb_rfc)
    adb_rfc_auc_scores.append(adb_rfc_auc_score)
    adb_rfc_scores_test.append(acc_adb_rfctest)
    
print("Train Scores list:", adb_rfc_scores)
print("Test Scores list:",adb_rfc_scores_test)
print("AUC list:",adb_rfc_auc_scores)


# In[138]:


sns.lineplot(x= estimators , y= adb_rfc_scores)


# In[140]:


sns.barplot(x= estimators , y= adb_rfc_scores_test)


# In[139]:


sns.lineplot(x= estimators , y= adb_rfc_scores_test)


# In[141]:


sns.barplot(x= estimators , y= adb_rfc_auc_scores)


# In[135]:


sns.lineplot(x= estimators , y= adb_rfc_auc_scores)


# In[156]:


"""GENEL KARŞILAŞTIRMALAR"""

classifiers=["Log Reg","KNN","NB","SVM","DTree","RForest","BGDT","ADBRF"]
results=[79.866,79.357,81.917,80.447,83.39,82.56,82.52,83.8]


# In[160]:


sns.set(style="whitegrid")
ax=sns.barplot(x= classifiers , y= results)


# In[168]:


sns.set(style="darkgrid")
ax= sns.lineplot(x= classifiers , y= results,sort=False)


# In[ ]:




