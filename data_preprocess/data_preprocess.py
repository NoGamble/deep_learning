import os
import pandas as pd

os.makedirs(os.path.join("..", "data"), exist_ok=True)
datafile = os.path.join("..", "data", "house_tiny.csv")

with open(datafile, "w") as f:
    f.write("NumRooms,Alley,Street,YearBuilt,Price\n")
    f.write("NA,Pave,Grvl,1990,127500\n")
    f.write("2,NA,Pave,2005,106000\n")
    f.write("4,NA,Pave,NA,178100\n")
    f.write("NA,NA,Grvl,1980,140000\n")
    f.write("3,Pave,Pave,2010,190000\n")
    f.write("NA,Grvl,NA,1995,135000\n")
    f.write("5,NA,Pave,2020,300000\n")
    f.write("2,Pave,Grvl,NA,98000\n")
