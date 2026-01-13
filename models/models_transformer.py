import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from itertools import product


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=SEQ_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerRegressor(nn.Module):
    def __init__(self, d_model=128, num_layers=2, n_head=4, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_enc(x)
        h = self.encoder(x)
        h_mean = h.mean(dim=1)
        h = self.norm(h_mean)
        h = self.dropout(h)
        out = self.head(h)
        return out.squeeze(-1)



