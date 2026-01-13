results = train_regressor("train.csv",
                                  D_MODEL=128,
                                  NUM_LAYERS=4,
                                  N_HEAD=4,
                                  LR=5e-4,
                                  DROPOUT=0.3,
                                  PATIENCE=10,
                                  EPOCHS=500)
torch.save(results['best_model_state'], "best_transformer.pt")


#inference on test.csv
def predict_on_test(
    test_path,
    model_state,
    output_path="test_pred.csv",
    D_MODEL=128,
    NUM_LAYERS=4,
    N_HEAD=4,
    DROPOUT=0.3,
    BATCH_SIZE=256
):
    """
    Run inference on test sequences and save predictions to CSV.
    """
    import pandas as pd

    print(f"Loading test data from {test_path}...")

    # Load dataset in inference mode
    test_dataset = ProteinActivityDataset(test_path, inference_mode=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Found {len(test_dataset)} sequences")

    print("Loading model...")
    model = TransformerRegressor(
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        n_head=N_HEAD,
        dropout=DROPOUT
    ).to(DEVICE)
    model.load_state_dict(model_state)
    model.eval()

    print("Running inference...")
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            pred = model(batch)
            predictions.extend(pred.cpu().numpy().flatten().tolist())

    test_df = pd.read_csv(test_path)

    output_df = pd.DataFrame({
        'seq': test_df['seq'],
        'activity': predictions
    })

    # Save to CSV
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    print(f"Prediction statistics:")
    print(f"   Mean: {output_df['activity'].mean():.4f}")
    print(f"   Std:  {output_df['activity'].std():.4f}")
    print(f"   Min:  {output_df['activity'].min():.4f}")
    print(f"   Max:  {output_df['activity'].max():.4f}")

    return output_df


