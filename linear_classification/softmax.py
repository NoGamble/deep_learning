import torch 
from IPython import display
from data_loader import load_data_fashion_mnist

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

batchsize = 256
num_inputs = 28 * 28
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

def soft_max(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

def net(X):
    return soft_max(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.astype(y.dtype) == y
    return float(cmp.astype(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metrics = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metrics.add(accuracy(net(X), y), y.numel())
    return metrics[0] / metrics[1]

    

def main():
    train_iter, test_iter = load_data_fashion_mnist(batchsize)
    X, y = next(iter(train_iter))
    # check the shape of train data and its labels
    print(X.shape, y.shape)    
    # X = torch.tensor([[1.0, 2.0, 3.], [4., 5., 6.]])
    # print(X)
    # print(X.sum(0, keepdim=True))


if __name__ == '__main__':
    main()
