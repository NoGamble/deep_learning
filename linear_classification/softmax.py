import torch 
from IPython import display
from data_loader import load_data_fashion_mnist

def soft_max(X):
    X_exp = torch.exp(X)
    
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition




def main():
    batchsize = 256
    train_iter, test_iter = load_data_fashion_mnist(batchsize)
    X, y = next(iter(train_iter))
    # check the shape of train data and its labels
    print(X.shape, y.shape)

    num_inputs = 28 * 28
    num_outputs = 10
    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)
    
    X = torch.tensor([[1.0, 2.0, 3.], [4., 5., 6.]])
    print(X)
    print(X.sum(1, keepdim=True))


if __name__ == '__main__':
    main()
