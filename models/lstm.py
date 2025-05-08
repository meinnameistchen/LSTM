# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(output)

class LSTMAttentionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes, output_dim, attention_dim, num_heads=4, dropout=0.2):

        super().__init__()

        self.num_layers = len(hidden_layer_sizes)
        self.lstm_layers = nn.ModuleList()

        self.lstm_layers.append(nn.LSTM(input_dim, hidden_layer_sizes[0], batch_first=True))


        for i in range(1, self.num_layers):
            self.lstm_layers.append(nn.LSTM(hidden_layer_sizes[i - 1], hidden_layer_sizes[i], batch_first=True))

        self.attention = MultiHeadAttention(attention_dim, num_heads)


        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)


        self.classifier = nn.Linear(hidden_layer_sizes[-1], output_dim)


        self.dropout = nn.Dropout(dropout)

    def forward(self, input_seq):

        batch_size = input_seq.size(0)
        seq_length = int((input_seq.size(1) ** 0.5))
        input_seq = input_seq.view(batch_size, seq_length, seq_length)

        lstm_out = input_seq
        for lstm in self.lstm_layers:
            lstm_out, _ = lstm(lstm_out)
            lstm_out = self.dropout(lstm_out)


        attention_features = self.attention(lstm_out)

        x = self.adaptive_pool(attention_features.permute(0, 2, 1))

        x = x.view(batch_size, -1)

        out = self.classifier(x)
        return out