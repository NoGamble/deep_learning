# tips: use vectorization for speed
# let's implement a solution to linear regeression from scratch

import torch
import pandas as pd
import numpy as np
import random

def synthetic_data(w, b, num_examples): 
    X = torch.normal(0, 1, (num_examples, len(w))) 
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, lables = synthetic_data(true_w, true_b, 1000)

print('features:', features[0], '\nlable:', lables[0])

def data_iter(batch_size, features, lables):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], lables[batch_indices]

batch_size= 10
for X, y in data_iter(batch_size, features, lables):
    print('X:', X, '\ny:', y)
    break

w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def linreg(X, w, b):
    return torch.matmul(X, w) + b
def square_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2 


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

lr = 0.03
num_epochs = 3
net = linreg
loss = square_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, lables):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_loss = loss(net(features, w, b), lables)
        print(f'epoch {epoch + 1}, loss {float(train_loss.mean()):f}')

print(f'w: {true_w - w.reshape(true_w.shape)}')
print(f'b: {true_b - b}')
