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

# operations
print(X + y)

X[0:2, :] = 1

print(X)

# save memory
mem_sav = torch.zeros_like(X)
print("id(mem_sav)", id(mem_sav))
mem_sav[:] = X + y
print("id(mem_sav)", id(mem_sav))

# convert to other object
A = X.numpy()
B = torch.tensor(A)
print(type(A), type(B))

# exercise for 2.1
a = torch.arange(3).reshape(3, 1)
print(a)
b = torch.arange(2).reshape(1, 2)
print(b)
print("a+b:", a + b)

A = torch.tensor([[[1, 1, 1]], [[2, 2, 2]]])

B = torch.tensor([[[1], [2], [3], [4]]])

result = A + B

print(result)

new_tensor = torch.randn(2, 3, 4)
print(new_tensor)
