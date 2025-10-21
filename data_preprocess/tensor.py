import torch

x = torch.arange(12)
# size of the tensor
print(x.numel())

# change the shape of the tensor
X = x.reshape(3, 4)
Y = torch.zeros(2, 3, 4)
Z = torch.ones((2, 3, 4))
print(X)
print(Y)
print(Z)


