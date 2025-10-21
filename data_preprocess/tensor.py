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

# to get a random tensor
X = torch.randn(3, 4)
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y

