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



fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['legend.fontsize'] = 10
ax.plot(iris_data[:,0],  iris_data[:,1], 'o', c=species, markersize=8, color='blue', alpha=0.5, label='Sepal')
ax.plot(iris_data[:,2], iris_data[:,2], '^', c=species, markersize=8, alpha=0.5, color='red', label='Petal')

plt.title('Samples for class 1 and class 2')
ax.legend(loc='upper right')

plt.show()