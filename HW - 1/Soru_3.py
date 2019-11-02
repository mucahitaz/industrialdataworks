import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from sklearn.metrics import pairwise as pw

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = np.array(iris.target, dtype=int)

normalized_data = preprocessing.normalize(X)

a = (pw.rbf_kernel(normalized_data,Y = None,gamma=None))
print(a)

