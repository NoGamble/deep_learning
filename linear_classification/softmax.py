import torch 
from IPython import display
from data_loader import load_data_fashion_mnist

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
