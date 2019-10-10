import pandas as pd

pd.set_option('display.max_columns', 25)

df = pd.read_csv (r'/home/administrator/PycharmProjects/Uni/data/Rainier_Weather.csv', encoding = "ISO-8859-1")   #read the csv file (put 'r' before the path string to address any special characters, such as '\'). Don't forget to put the file name at the end of the path + ".csv"
# print(df.head())

to_drop= ['Wind Direction AVG']

df.drop(to_drop , inplace = True, axis=1)

print(df.head())