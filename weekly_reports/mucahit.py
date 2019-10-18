import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from sklearn.decomposition import PCA


plt.close('all')

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 1000)

df = pd.read_csv(r'/home/administrator/PycharmProjects/Uni/weekly_reports/data/Video_Games_Sales_as_at_22_Dec_2016.csv',
                 encoding="ISO-8859-1")

'''Başlangıçta veri setini incelemek ve fikir edinmek için kullanılabilecek komutlar'''


def StarterFunc():
    print(df.head())  # İlk 5 satır görüntülenir.


'''Eksik değer içeren kayıtların silinmesi yöntemi'''


def DropMissingValues():
    print("Amount of NaN values before execution\n", df.isnull().sum())
    df.dropna(how='any',
              inplace=True)  # Buradaki inplace kısmı gerçek veri üzerinde etkide bulunup bulunmama kararıdır.
    print("Amount of NaN values after execution\n", df.isnull().sum())
    return df  # Sutunlardaki toplam boş veri sayıları öğrenilir.


'''Eksik değer içeren verilerin medyan yardımı ile doldurulması'''


def MedianFillingFunc(self):
    df[self] = pd.to_numeric(df[self], errors='coerce')  # str verilerini NaN Formatına dönüştürür.
    median = df[self].median()
    df[self].fillna(median, inplace=True)
    print("Median of {} column is".format(self), median)
    return df


def ModeFillingFunc(self):
    mode = df[self].mode()
    df[self].fillna(mode, inplace=True)
    print("Mode of {} column is".format(self), mode)
    return df

normalized_data = 0


DropMissingValues()

df1 = df.iloc[:,5:8]
print(df1)

normalized_data = preprocessing.normalize(df1)
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
print("Normalized data r: \n" ,normalized_data_r)

print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))
