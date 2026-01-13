import itertools
import torch
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
from torch import nn
import pandas as pd
from torch.utils.data import Dataset
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB_SIZE = 20       # amino acids
SEQ_LEN = 12
EMBED_AA = 16
EMBED_S = 8


AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AA)}

class ProteinActivityDataset(Dataset):
    def __init__(self, csv_path, require_activity=True):
        df = pd.read_csv(csv_path)
        self.x = [[AA_TO_IDX[a] for a in seq] for seq in df['seq']]
        if require_activity:
            self.s = torch.tensor(df['activity'].values, dtype=torch.float32)
            self.mean = self.s.mean()
            self.std = self.s.std()
            self.s = (self.s - self.mean) / (self.std + 1e-8)
        else:
            self.s = None

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.long)
        if self.s is None:
            return x
        return x, self.s[idx]

    def __len__(self):
        return len(self.x)

