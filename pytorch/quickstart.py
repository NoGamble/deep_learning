import torch
from torch.utils.data import Dataset
from torchvision import datasets

class Timer:
    def __init__(self):
        self.times = []
        self.start()