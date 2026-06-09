import torch
import numpy as np
import matplotlib as plt 

def f(x):
    return 3 * x ** 2 - 4 * x

x = torch.arange(4.0, requires_grad=True)
print('x:', x)

y = 2 * torch.dot(x, x)
print('y:', y)

y.backward()
# can not y.backward() again, cause there's no mid-result left in mem
print('x.grad', x.grad)

x.grad.zero_()
y = x.sum()
y.backward()
print('x_sum.grad', x.grad)

# Python Cotrol Flow
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True, dtype=torch.float64)
d = f(a)
d.backward()
print('python flow verification', a.grad == d / a)

