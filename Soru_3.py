import numpy as np
from sklearn import datasets
from sklearn.gaussian_process.kernels import RBF
from sklearn import preprocessing
from sklearn.metrics import pairwise as pw
from scipy.spatial.distance import pdist, squareform
import scipy

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = np.array(iris.target, dtype=int)

normalized_data = preprocessing.normalize(X)

a = (pw.rbf_kernel(normalized_data,Y = None,gamma=None))
print(a)

