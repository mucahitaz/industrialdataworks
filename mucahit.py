import pandas as pd

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 1000)

df = pd.read_csv (r'/home/administrator/PycharmProjects/Uni/data/Video_Games_Sales_as_at_22_Dec_2016.csv', encoding = "ISO-8859-1")

'''Başlangıçta veri setini incelemek ve fikir edinmek için kullanılabilecek komutlar'''

def StarterFunc():
    print(df.head())  #İlk 5 satır görüntülenir.


'''Eksik değer içeren kayıtların silinmesi yöntemi'''

def DropMissingValues():
    print("Amount of NaN values before execution\n", df.isnull().sum())
    df.dropna(how='any' , inplace= True) #Buradaki inplace kısmı gerçek veri üzerinde etkide bulunup bulunmama kararıdır.
    print ("Amount of NaN values after execution\n" , df.isnull().sum())
    return df # Sutunlardaki toplam boş veri sayıları öğrenilir.


'''Eksik değer içeren verilerin medyan yardımı ile doldurulması'''

def MedianFillingFunc(self):
    median = df[self].median()
    df[self].fillna(median, inplace=True)
    print("Median of {} column is".format(self) , median)
    return df

print(DropMissingValues())
print(MedianFillingFunc())  #tbd value problem çıkarıyor.