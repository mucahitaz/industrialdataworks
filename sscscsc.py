import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_iris
import numpy as np

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)



dataset = load_iris(return_X_y=False)

iris=load_iris()
x = iris.data
y= iris.target

# # print(dataset.DESCR)
# # print(dataset)
#
# df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
# dataset.feature_names
#
# df['target_names'] = np.take(dataset.target_names, dataset.target)
#
# # print(df.head(50))
# # print(df)
# # print(dataset.groupby('class').size())
#
# print(pd.DataFrame(dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)))
# print(plt.show())