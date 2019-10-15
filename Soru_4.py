import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn import datasets


iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names


kpca = KernelPCA(n_components=2, kernel="rbf")
X_kpca = kpca.fit_transform(X)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5),
            (0.4, 0.6, 0), (0.6, 0.4, 0), (0, 0.6, 0.4), (0.5, 0.3, 0.2),)
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    position = y == i
    plt.scatter(X_kpca[position, 0], X_kpca[position, 1], color=color, alpha=.8, label=target_name)

ax.set_xlabel("Sepal Length")
ax.set_ylabel("Sepal Width")
ax.legend(loc="best")
ax.set_title("KPca")
plt.show()