import numpy as np
from sklearn.datasets import load_iris
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d


iris_obj = load_iris()
iris_data = iris_obj.data
species = iris_obj.target
print(species)
print(iris_obj.target_names)



# mds = MDS(n_components=2)

plt.scatter(iris_data[:,0], iris_data[:,1], c=species , cmap=plt.cm.brg)
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.show()

plt.scatter(iris_data[:,2], iris_data[:,3], c=species,cmap=plt.cm.brg)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()

# similarities = euclidean_distances(iris_data.astype(np.float64))
# print (np.abs(similarities - similarities.T).max())
#
# mds.fit(iris_data.astype(np.float64))
#
# similarities = euclidean_distances(iris_data.astype(np.float32))
# print (np.abs(similarities - similarities.T).max())
#
# mds.fit(iris_data.astype(np.float32))

# pos = mds.fit(similarities).embedding_