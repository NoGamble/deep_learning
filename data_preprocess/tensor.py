import torch

# tensor basic knowledges
x = torch.arange(12)

print(x)
print(x.shape)
print(x.size)
print(x.numel())

X = x.reshape(3, 4)

print(X)
print(X.shape)
print(X.size)
print(X.numel())

y = torch.zeros([3, 4])
print(y)

z = torch.ones((2, 3, 4))
print(z)
