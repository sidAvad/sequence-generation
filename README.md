# Protein Sequence Modeling for Activity Prediction and Conditional Generation

**Author:** Siddharth Avadhanam  

This repository implements an end-to-end protein sequence modeling pipeline combining **predictive modeling** and **conditional generation**, suitable for research and industry protein design workflows.

---

## Overview

The project addresses three core needs in protein engineering:

- Accurate **sequence → activity prediction**
- **Generative modeling** of novel protein sequences
- Quantitative evaluation of generated sequences without ground-truth labels

This is achieved by pairing a high-quality Transformer regressor with a Conditional Variational Autoencoder (CVAE) and evaluating generation quality in a closed loop.

---

## Key Results

- **Transformer regressor**
  - Pearson correlation ≈ **0.86** on a held-out test set
  - Stable performance under 5-fold cross-validation
- **CVAE generator**
  - Produces novel sequences enriched for high-activity motifs
  - Highlights known limitations of weakly conditioned VAEs

---

## Repository Structure

├── train.csv
├── test.csv
├── test_pred.csv
├── best_transformer.pt
├── generated_seqs2.csv
├── generated_pred.csv

---

## Part I — Activity Prediction

### Model

A Transformer encoder regressor with:
- Amino acid embeddings + sinusoidal positional encoding
- 4 stacked Transformer encoder blocks (4 heads each)
- Mean pooling over sequence positions
- Layer normalization, dropout, and linear regression head

### Training

- 20% held-out test split
- 5-fold cross-validation on remaining data
- AdamW optimizer, MSE loss
- Early stopping and dropout
- Grid search over depth, heads, learning rate, and dropout

### Evaluation

- Mean Squared Error (MSE)
- **Pearson correlation** (primary metric)
- Spearman rank correlation

Correlation is emphasized for ranking sequences by activity.

---

## Part II — Conditional Sequence Generation

### Model

A Conditional Variational Autoencoder (CVAE):
- Encoder maps sequence + activity → latent distribution
- Decoder maps latent code + activity → amino acid logits
- KL-regularized latent space with annealing for stability

### Generation

- Sample latent variables from a standard normal
- Condition on target activity
- Sample amino acids with optional temperature scaling

---

## Closed-Loop Validation

Generated sequences are evaluated by the Transformer regressor:

- High-activity conditioning produces recognizable motifs
- Average predicted activity ≈ **4.8** for target activity = 7

This indicates partial alignment and motivates stronger conditioning strategies.

---

## Limitations & Future Work

- Weak conditioning of latent prior in standard CVAEs
- Potential improvements:
  - Conditional latent priors
  - Auxiliary prediction heads
  - Conditional Transformer decoders
  - Regressor-guided latent optimization

---

## Intended Use

Designed for:
- Protein design research
- Generative model benchmarking
- Integration into active learning pipelines

The framework generalizes to other sequence–property modeling problems with minimal modification.
