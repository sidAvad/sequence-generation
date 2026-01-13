import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from itertools import product

# generate new samples
def generate(dec, ds, activities, num_samples=100, Z_DIM=8, temperature=1.0):
    dec.eval()
    acts = (torch.tensor(activities) - ds.mean) / (ds.std + 1e-8)
    results = []
    with torch.no_grad():
        for a in acts:
            for _ in range(num_samples):
                z = torch.randn(1, Z_DIM).to(DEVICE)
                logits = dec(z, a.unsqueeze(0).to(DEVICE))

                probs = torch.softmax(logits / temperature, dim=-1)
                idx = torch.multinomial(probs.view(-1, probs.size(-1)), 1).squeeze()

                seq = ''.join(AA[i] for i in idx)
                results.append((a.item(), seq))
    return results


if __name__ == "__main__":
    Z_DIM = 16
    enc, dec, ds = train_cvae("train.csv", H_DIM=128, Z_DIM=Z_DIM, LR=0.001,
                              DROPOUT=0.1, EPOCHS=500, KL_ANNEAL_STEPS=5000)
    samples = generate(dec, ds, [7], Z_DIM=Z_DIM, temperature=1.0)
    for a, seq in samples:
        print(f"{a:.2f}\t{seq}"


	predict_on_test(
    	"generated_seqs2.csv",
    	results["best_model_state"],
    	output_path="generated_pred.csv")


