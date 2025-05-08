# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
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

      
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
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

       
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  
        )

       
        self.feature_fusion = nn.Linear(hidden_layer_sizes[-1] + 128, hidden_layer_sizes[-1])
        self.classifier = nn.Linear(hidden_layer_sizes[-1], output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.size(0), input_seq.size(1)

       
        if input_seq.dim() == 2:
            side = int(seq_len ** 0.5)
            assert side * side == seq_len, "Input is not square and can't be reshaped into square"
            input_seq = input_seq.view(batch_size, side, side)

     
        cnn_input = input_seq.permute(0, 2, 1)  
        cnn_features = self.cnn(cnn_input).squeeze(-1) 

       
        lstm_out = input_seq
        for lstm in self.lstm_layers:
            lstm_out, _ = lstm(lstm_out)
            lstm_out = self.dropout(lstm_out)

    
        attn_out = self.attention(lstm_out)
        pooled = self.adaptive_pool(attn_out.permute(0, 2, 1)).squeeze(-1)  # (B, D)

       
        fused = torch.cat((pooled, cnn_features), dim=1)
        fused = self.feature_fusion(fused)

       
        out = self.classifier(fused)
        return out
