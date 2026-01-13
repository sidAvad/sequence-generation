import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from itertools import product
from scipy.stats import pearsonr, spearmanr

def train_regressor(
    train_path,
    D_MODEL=128,
    NUM_LAYERS=2,
    N_HEAD=4,
    LR=1e-4,
    DROPOUT=0.1,
    EPOCHS=100,
    BATCH_SIZE=256,
    PATIENCE=10,
    MIN_DELTA=1e-4,
    N_FOLDS=5
):
    """
    Train transformer regressor with k-fold cross-validation and early stopping.
    """
    from sklearn.model_selection import KFold

    # load full dataset
    full_ds = ProteinActivityDataset(train_path)

    # split off test set
    test_size = int(0.2 * len(full_ds))
    train_val_size = len(full_ds) - test_size
    train_val_ds, test_ds = random_split(full_ds, [train_val_size, test_size])

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # k-fold cross-validation
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_results = []
    fold_models = []

    print(f"Starting {N_FOLDS}-fold cross-validation...")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(train_val_ds)))):
        print(f"\n FOLD {fold + 1}/{N_FOLDS}")

        train_subset = torch.utils.data.Subset(train_val_ds, train_idx)
        val_subset = torch.utils.data.Subset(train_val_ds, val_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)

        model = TransformerRegressor(
            d_model=D_MODEL,
            num_layers=NUM_LAYERS,
            n_head=N_HEAD,
            dropout=DROPOUT
        ).to(DEVICE)

        opt = torch.optim.AdamW(model.parameters(), lr=LR)

        best_val = float("inf")
        best_state = None
        patience_counter = 0

        #training loop for each fold
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                loss = F.mse_loss(pred, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item() * x.size(0)
            train_loss = total_loss / len(train_subset)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    pred = model(x)
                    loss = F.mse_loss(pred, y)
                    val_loss += loss.item() * x.size(0)
            val_loss /= len(val_subset)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d} | train={train_loss:.4f} | val={val_loss:.4f}")

            # early stopping
            if val_loss + MIN_DELTA < best_val:
                best_val = val_loss
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f" Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
                break

        fold_results.append(best_val)
        fold_models.append(best_state)
        print(f" Fold {fold + 1} best val loss: {best_val:.4f}")

    #cross-val results
    print("CROSS-VALIDATION RESULTS")
    mean_val = sum(fold_results) / len(fold_results)
    std_val = (sum((x - mean_val) ** 2 for x in fold_results) / len(fold_results)) ** 0.5
    print(f"Mean val loss: {mean_val:.4f} ± {std_val:.4f}")
    for i, val_loss in enumerate(fold_results):
        print(f"  Fold {i+1}: {val_loss:.4f}")

    #evaluate best model on test set
    best_fold_idx = fold_results.index(min(fold_results))
    best_model_state = fold_models[best_fold_idx]

    model = TransformerRegressor(
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        n_head=N_HEAD,
        dropout=DROPOUT
    ).to(DEVICE)
    model.load_state_dict(best_model_state)
    model.eval()

    test_loss = 0.0
    predictions = []
    true_values = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            predictions.extend(pred.cpu().numpy().flatten())
            true_values.extend(y.cpu().numpy().flatten())
            loss = F.mse_loss(pred, y)
            test_loss += loss.item() * x.size(0)
    test_loss /= len(test_ds)
    predictions = np.array(predictions)
    true_values = np.array(true_values)

    # compute correlations
    pearson_corr, pearson_pval = pearsonr(true_values, predictions)
    spearman_corr, spearman_pval = spearmanr(true_values, predictions)

    # final results
    print(f"\n Best model from fold {best_fold_idx + 1}")
    print(f" Final test loss: {test_loss:.4f}")
    print("=" * 60)
    print("TEST SET CORRELATION")
    print(f"Pearson r:  {pearson_corr:.4f} (p={pearson_pval:.2e})")
    print(f"Spearman ρ: {spearman_corr:.4f} (p={spearman_pval:.2e})")


    return {
        'mean_val_loss': mean_val,
        'std_val_loss': std_val,
        'fold_results': fold_results,
        'test_loss': test_loss,
        'pearson_corr': pearson_corr,
        'pearson_pval': pearson_pval,
        'spearman_corr': spearman_corr,
        'spearman_pval': spearman_pval,
        'best_fold': best_fold_idx + 1,
        'best_model_state': best_model_state
    }
