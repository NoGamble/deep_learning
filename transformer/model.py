import torch
import torch.nn as nn
import math

# This is a implementation of the core of transformer according to hkproj's repo

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwadNeetwork(nn.Module):
    def __init__(self, d_model: int, d_ff: int, drop_out: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(drop_out) # w2 and b2
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) -> (batch, seq_len, d_model) 
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: int) -> None:
        super().__init__()

class ResidualConnection(nn.Module):
    def __init__(self, features: int, droupout: float) -> None:
        super().__init__()
        