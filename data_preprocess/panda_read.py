import pandas as pd
import os
import torch

datafile = os.path.join("..", "data", "house_tiny.csv")

data = pd.read_csv(datafile)

print(data)

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean(numeric_only=True))
# print(inputs)

# X = torch.tensor(inputs.to_numpy(dtype=float))
# Y = torch.tensor(outputs.to_numpy(dtype=float))
# print(X)
# print(Y)

# exercise
null_counts = data.isnull().sum()
most_missing_col = null_counts.idxmax()

data_mod = data.drop(columns=[most_missing_col])
print(data_mod)
