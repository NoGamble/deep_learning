import torch

# scalars
x = torch.tensor(3.0)
y = torch.tensor(2.0)

# vectors
z = torch.tensor([[1, 1], [2, 2]])
print(z.shape)
print(len(z))

# Metrix
A = torch.arange(6).reshape(3, 2)
print(A)
print(A.T)

# Tensors

# Reduction
print(A.sum())
print(A.sum(dim=0))
print(A.sum(dim=0).shape)
print(A.sum(dim=1))
print(A.sum(dim=1).shape)

# Non-Reduction Sum
print(A.sum(dim=1, keepdim=True))
print(A.sum(dim=1, keepdim=True).shape)

# Dot Product
x = torch.arange(3, dtype=torch.float32)
y = torch.ones(3, dtype=torch.float32)
print("x:", x)
print("y:", y)
print("x dot y:", torch.dot(x, y))
print(torch.dot(x, y) == torch.sum(x * y))

# Metrix-Vector Product
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print(A.shape)
print(A)
x = torch.ones(4, dtype=torch.float32)
print(x.shape)
print(x)
print(torch.mv(A, x))
print(torch.mv(A, x).shape)

# Metrix-Metrix
B = torch.ones(4, 3)
print(torch.mm(A, B))

# Norm
u = torch.tensor([3.0, -4.0])
print(torch.norm(u))
