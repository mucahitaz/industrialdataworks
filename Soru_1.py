#!/usr/bin/env python
# coding: utf-8

# In[23]:


import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import preprocessing
import numpy as np


# In[21]:


iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names
print(target_names)


# In[ ]:


print(X)


# In[ ]:


normalized_data = preprocessing.normalize(X)
print(normalized_data)


# In[26]:


mean_vec = np.mean(normalized_data, axis=0)
cov_mat = (normalized_data - mean_vec).T.dot((normalized_data - mean_vec)) / (normalized_data.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)


# In[27]:


cov_mat = np.cov(normalized_data.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


# In[28]:


pca = PCA(n_components=2)
normalized_data_r = pca.fit(normalized_data).transform(normalized_data)
print(normalized_data_r)


# In[19]:


print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2


# In[20]:


for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(normalized_data_r[y == i, 0], normalized_data_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')
plt.show()

# In[ ]:




