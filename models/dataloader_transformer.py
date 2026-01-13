import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from itertools import product

#CONFIG
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AA)}
VOCAB_SIZE = len(AA)
SEQ_LEN = 12
BATCH_SIZE = 256
EPOCHS = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu"


class ProteinActivityDataset(Dataset):
    def __init__(self, csv_path, inference_mode=False):
        """
        Dataset for protein sequences with optional activity values.
        """
        df = pd.read_csv(csv_path)
        self.x = [[AA_TO_IDX[a] for a in seq] for seq in df['seq']]
        self.inference_mode = inference_mode

        if not inference_mode:
            self.s = torch.tensor(df['activity'].values, dtype=torch.float32)
            self.mean = self.s.mean()
            self.std = self.s.std()
            self.s = (self.s - self.mean) / (self.std + 1e-8)
        else:
            self.mean = None
            self.std = None
            self.s = None

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.long)
        if self.inference_mode:
            return x
        else:
            return x, self.s[idx]

    def __len__(self):
        return len(self.x)
