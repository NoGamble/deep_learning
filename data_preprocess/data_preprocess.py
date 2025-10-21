import os
import pandas as pd

# the file path
os.makedirs(os.path.join(".", 'data'), exist_ok = True)
data_file_path = os.path.join(".", "data", "house_tiny.csv")

with open(data_file_path, 'w') as f:
    # column names
    f.write('NumRooms,Alley,Price\n')
    # writting into data:
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

# Handling the missing values
data = pd.read_csv(data_file_path)
print(data)

# iloc is pandas integer location
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs['NumRooms'] = inputs['NumRooms'].fillna(inputs['NumRooms'].mean())
# One-hot coding
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)


