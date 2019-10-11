import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from sklearn.decomposition import PCA as sklearnPCA

#https://sebastianraschka.com/Articles/2015_pca_in_3_steps.html
#https://towardsdatascience.com/spectral-clustering-aba2640c0d5b
#https://www.kaggle.com/lambdaofgod/kernel-pca-examples#KPCA-of-circles
#https://scikit-learn.org/stable/auto_examples/decomposition/plot_kernel_pca.html
#https://medium.com/@ekrem.hatipoglu/machine-learning-classification-support-vector-machine-kernel-trick-part-10-7ab928333158
#https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
#https://medium.com/@belen.sanchez27/predicting-iris-flower-species-with-k-means-clustering-in-python-f6e46806aaee



pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)


df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None,
    sep=',')

df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True) # drops the empty line at file-end
print(df)

X = df.iloc[:,0:4].values
y = df.iloc[:,4].values

# print(X)
# print(y)

label_dict = {1: 'Iris-Setosa',
              2: 'Iris-Versicolor',
              3: 'Iris-Virgnica'}

feature_dict = {0: 'sepal length [cm]',
                1: 'sepal width [cm]',
                2: 'petal length [cm]',
                3: 'petal width [cm]'}

with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(8, 6))
    for cnt in range(4):
        plt.subplot(2, 2, cnt+1)
        for lab in ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'):
            plt.hist(X[y==lab, cnt],
                     label=lab,
                     bins=10,
                     alpha=0.3,)
        plt.xlabel(feature_dict[cnt])
    plt.legend(loc='upper right', fancybox=True, fontsize=8)

    # plt.tight_layout()
    # plt.show()

# X_std = StandardScaler().fit_transform(X)
X_normalized = preprocessing.normalize(X, norm='l2')

mean_vec = np.mean(X_normalized, axis=0)
cov_mat = (X_normalized - mean_vec).T.dot((X_normalized - mean_vec)) / (X_normalized.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

for ev in eig_vecs.T:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
print('Everything ok!')


# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),
                      eig_pairs[1][1].reshape(4,1)))

Y = X_normalized.dot(matrix_w)

# with plt.style.context('seaborn-whitegrid'):
#     plt.figure(figsize=(6, 4))
#     for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
#                         ('blue', 'red', 'green')):
#         plt.scatter(Y[y==lab, 0],
#                     Y[y==lab, 1],
#                     label=lab,
#                     c=col)
#     plt.xlabel('Principal Component 1')
#     plt.ylabel('Principal Component 2')
#     plt.legend(loc='lower center')
#     plt.tight_layout()
#     plt.show()
