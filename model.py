import torch
import torch.nn as nn
import math

# d_model = dimension of model (Embedding dimension -> Each word is represented say as a 4 dimensional vector)
# vocab_size = number of words in our dictionary 
# embeddings_example.py

class InputEmbeddings(nn.Module):
    def __init__(self, d_model : int, vocab_size : int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,  d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len  #"Your cat is the best cat"
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len * d_model)  refer examples.ipynb
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of len = seq_len to represent position of a word inside sentence 
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/ d_model))
        # Sin for even positions 
        pe[:, 0::2] = torch.sin(position * div_term)
        # Cosine for odd positions 
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False)  #To make the model not to learn the positions as they are a one time thing
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))   #parameter is used to make it learnable -> Multiplicative
        self.bias = nn.Parameter(torch.zeros(1))   #Additive

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff : int, dropout : float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) #W1 and B1 (bias is by default true)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) #W2 and B2

    def forward(self, x):
        # (Batch, seq_len, d_model) -> (Batch, d_model, d_ff) -> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
    











