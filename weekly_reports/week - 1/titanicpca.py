import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.decomposition import PCA
from sklearn import preprocessing
import tensorflow as tf

from subprocess import check_output
print(check_output(["ls", "/home/mucahit/PycharmProjects/school/industrialdataworks/weekly_reports/data/titanic"]).decode("utf8"))

train_data = pd.read_csv('/home/mucahit/PycharmProjects/school/industrialdataworks/weekly_reports/data/titanic/train.csv', dtype={'Age': np.float16})
test_data = pd.read_csv('/home/mucahit/PycharmProjects/school/industrialdataworks/weekly_reports/data/titanic/test.csv')

train_data.head()

print('train size: %d, test size: %d' % (train_data.size, test_data.size))

nans = {}
for colname in train_data.columns:
    nans[colname] = train_data[train_data[colname].isnull()].size
print(nans)

drop_column = ['Cabin', 'Ticket' ]
train_data.drop(drop_column, axis=1, inplace = True)

train_data['Age'].fillna(train_data['Age'].median(), inplace = True)

train_features = ['Age', 'Sex_number', 'Pclass']
train_data['Sex_number'] = train_data.apply(lambda row: 0 if row['Sex'] == 'male' else 1, axis=1)
train_X = train_data[train_features].as_matrix()
train_Y = train_data.Survived.as_matrix()
train_data[train_features].head()
print(train_Y)


normalized_data = preprocessing.normalize(train_X)
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

pca.fit(normalized_data_r)
trans = pca.transform(normalized_data_r)

fig, axs = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=False)


female_trans = np.array([tran for is_female, tran in zip(train_data['Sex_number'], trans) if is_female==1])
axs[0, 0].plot(trans[:,0], trans[:,1], '.', label='Male')
axs[0, 0].plot(female_trans[:,0], female_trans[:,1], 'r.', label='Female')
axs[0, 0].set_title('PCA with use of Sex (M / F)')
axs[0, 0].legend()

plt.show()