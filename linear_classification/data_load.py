import torch
import numpy as np
import torchvision
from torch.utils import data
from torchvision import transforms

trans = transforms.ToTensor()

mnist_train = torchvision.datasets.FashionMNIST(
    root='../data', train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root='../data', train=False, transform=trans, download=True
)

# check the size of the whole datasets
print(len(mnist_train), len(mnist_test))

# check the size of each pic
print(mnist_train[0][0].shape)
