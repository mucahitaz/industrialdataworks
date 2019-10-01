import pandas as pd

df = pd.read_csv (r'/home/administrator/PycharmProjects/Uni/drivers.csv', encoding = "ISO-8859-1")   #read the csv file (put 'r' before the path string to address any special characters, such as '\'). Don't forget to put the file name at the end of the path + ".csv"
print (df)

