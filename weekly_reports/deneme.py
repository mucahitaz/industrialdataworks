import sys
import pandas as pd
import matplotlib
import numpy as np
import scipy as sp
import sklearn
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 1000)

data_raw = pd.read_csv('/home/administrator/PycharmProjects/Uni/weekly_reports/data/titanic/train.csv')

y = data_raw.Survived.values

drop_column = ['PassengerId', 'Cabin', 'Ticket','Name','Fare','Embarked','SibSp','Parch','Survived','Sex']
data_raw.drop(drop_column, axis=1, inplace=True)

print(data_raw)

data_cleaner = [data_raw]

for dataset in data_cleaner:
    # complete missing age with median
    dataset['Age'].fillna(dataset['Age'].median(), inplace=True)

total = [data_raw]
for dataset in total:
    dataset.loc[dataset['Age'] <= 18, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

# dunno = [data_raw]
# for dataset in dunno:
#     dataset.loc[dataset['Sex'] == 'Male', 'Sex'] = 0
#     dataset.loc[dataset['Sex'] == 'Female' , 'Sex'] = 1

normalized_data = preprocessing.normalize(data_raw)
print(normalized_data)

mean_vec = np.mean(normalized_data, axis=0)
cov_mat = (normalized_data - mean_vec).T.dot((normalized_data - mean_vec)) / (normalized_data.shape[0] - 1)
print('Covariance matrix \n%s' % cov_mat)

cov_mat = np.cov(normalized_data.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' % eig_vecs)
print('\nEigenvalues \n%s' % eig_vals)

pca = PCA(n_components=2)
normalized_data_r = pca.fit(normalized_data).transform(normalized_data)
print(normalized_data_r)

print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['navy', 'turquoise']
lw = 2

for color, i, target_name in zip(colors, [0, 1], y):
    plt.scatter(normalized_data_r[y == i, 0], normalized_data_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')
plt.show()
