import tkinter as tk
from tkinter import filedialog
import pandas as pd


pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 1000)


root = tk.Tk()

canvas1 = tk.Canvas(root, width=300, height=300, bg='lightsteelblue2', relief='raised')
canvas1.pack()


def getCSV():
    global df
    import_file_path = filedialog.askopenfilename()
    data = pd.read_csv(import_file_path,encoding = "ISO-8859-1")
    df = pd.DataFrame(data)
    print(df)


browseButton_CSV = tk.Button(text="      Import CSV File     ", command=getCSV, bg='green', fg='white',
                             font=('helvetica', 12, 'bold'))
canvas1.create_window(150, 150, window=browseButton_CSV)

root.mainloop()