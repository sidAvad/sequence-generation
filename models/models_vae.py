import itertools
import torch
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
from torch import nn
import pandas as pd
from torch.utils.data import Dataset
import torch


class Encoder(nn.Module):
    def __init__(self, H_DIM, Z_DIM, dropout=0.3):
        super().__init__()
        self.embed_x = nn.Embedding(VOCAB_SIZE, EMBED_AA)
        self.embed_s = nn.Sequential(nn.Linear(1, EMBED_S), nn.ReLU())
        self.net = nn.Sequential(
            nn.Linear(SEQ_LEN * EMBED_AA + EMBED_S, H_DIM),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H_DIM, H_DIM),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.mu = nn.Linear(H_DIM, Z_DIM)
        self.logvar = nn.Linear(H_DIM, Z_DIM)

    def forward(self, x, s):
        x = self.embed_x(x).view(x.size(0), -1)
        s = self.embed_s(s.unsqueeze(1))
        h = self.net(torch.cat([x, s], dim=1))
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, H_DIM, Z_DIM, dropout=0.3):
        super().__init__()
        self.embed_s = nn.Sequential(nn.Linear(1, EMBED_S), nn.ReLU())
        self.net = nn.Sequential(
            nn.Linear(Z_DIM + EMBED_S, H_DIM),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H_DIM, SEQ_LEN * VOCAB_SIZE)
        )

    def forward(self, z, s):
        s = self.embed_s(s.unsqueeze(1))
        h = self.net(torch.cat([z, s], dim=1))
        return h.view(-1, SEQ_LEN, VOCAB_SIZE)

#loss functions
def reparam(mu, logvar):
    std = torch.exp(0.5 * logvar)
    return mu + std * torch.randn_like(std)

def kl_normal(mu, logvar):
    return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar, dim=1)

def recon_loss(logits, x):
    return F.cross_entropy(logits.transpose(1, 2), x, reduction='none').sum(dim=1)
