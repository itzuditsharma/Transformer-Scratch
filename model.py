import torch
import torch.nn as nn
import math

# d_model = dimension of model (Embedding dimension -> Each word is represented say as a 4 dimensional vector) - 512 in our demo
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
    
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)
        assert d_model % h == 0, "d_model in not divisible by h"
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)    # Practically, dk = dv and & h * dv = d_model

    # Attention 
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # (batch, h, seq_len, dk) -> (batch, h, seq_len, seq_len)
        # since (seqlen * dk) * (dk * seq_len) = (seq_len * seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) #(seq_len * seq_len)
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1)   #(batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores  #this is for visualization


    # mask is used when you want some words to not to interact with other words so we mask them
    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        # divide Q, K, V into smaller matrices to provide them to smaller heads 

        # ----Getting the smaller Matrices---- 
        # This way each head will see the entire sentence but only a smaller part of the embedding 
        # (Batch, seq_len, d_model)  --> (Batch, seq_len, h, d_k) --> (Batch, h, seq_len, d_k) 
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        # Calculate Attention 
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Combining all the small matrices to form H
        # (Batch, h, seq_len, d_k)  --> (Batch, seq_len, h, d_k) --> (batch, seq_len, d_model))
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    




        
























