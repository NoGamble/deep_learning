import os

# the file path
os.mkdir(os.path.join(".", 'data'), exit_ok = True)
data_file_path = os.path.join("..", "data", "house_tiny.csv")

with open(data_file_path, 'w') as f:
    # column names
    f.write('NumRooms,Alley,Price\n')
    # writting into data:
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

