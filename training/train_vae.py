import itertools
import torch
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
from torch import nn
import pandas as pd
from torch.utils.data import Dataset
import torch
 
def train_cvae(train_path, H_DIM, Z_DIM, LR, DROPOUT, KL_ANNEAL_STEPS=2000, EPOCHS=100, BATCH_SIZE=256, PATIENCE=10):
    full_ds = ProteinActivityDataset(train_path)
    val_size = int(0.2 * len(full_ds))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    enc = Encoder(H_DIM, Z_DIM, DROPOUT).to(DEVICE)
    dec = Decoder(H_DIM, Z_DIM, DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=LR)

    best_val_loss = float('inf')
    patience_counter = 0
    best_enc_state = None
    best_dec_state = None
    global_step = 0

    for epoch in range(EPOCHS):
        enc.train(); dec.train()
        train_loss_total = 0

        for x, s in train_loader:
            x, s = x.to(DEVICE), s.to(DEVICE)
            mu, logvar = enc(x, s)
            z = reparam(mu, logvar)
            logits = dec(z, s)
            rec = recon_loss(logits, x)
            kl = kl_normal(mu, logvar)
            kl_weight = min(1.0, global_step / KL_ANNEAL_STEPS)
            loss = (rec + kl_weight * kl).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss_total += loss.item() * x.size(0)
            global_step += 1

        train_loss = train_loss_total / len(train_ds)

        # validation
        enc.eval(); dec.eval()
        val_loss_total = 0
        with torch.no_grad():
            for x, s in val_loader:
                x, s = x.to(DEVICE), s.to(DEVICE)
                mu, logvar = enc(x, s)
                z = reparam(mu, logvar)
                logits = dec(z, s)
                rec = recon_loss(logits, x)
                kl = kl_normal(mu, logvar)
                val_loss_total += (rec + kl).mean().item() * x.size(0)

        val_loss = val_loss_total / len(val_ds)

        # early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_enc_state = enc.state_dict().copy()
            best_dec_state = dec.state_dict().copy()
            print(f"Epoch {epoch+1}: New best val_loss = {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}: val_loss = {val_loss:.4f} (patience {patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                enc.load_state_dict(best_enc_state)
                dec.load_state_dict(best_dec_state)
                break

    return enc, dec, full_ds


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
        print(f"{a:.2f}\t{seq}")
