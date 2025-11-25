import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, TensorDataset

# Import the shared LSTM model & config from client.py
from client import LSTMModel, TIMESTEPS, FEATURES, GLOBAL_NUM_CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_member_dataset(member_id: int):
    """
    Load full Member{member_id} dataset from ../../splits,
    reshape to (N, TIMESTEPS, FEATURES), normalize, return DataLoader.
    """
    base_path = os.path.join("..", "..", "splits")
    X_path = os.path.join(base_path, f"Member{member_id}_X.csv")
    y_path = os.path.join(base_path, f"Member{member_id}_Y.csv")

    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)

    # y as integer labels
    y_np = y.values.ravel().astype(int)

    # reshape X -> (N, TIMESTEPS, FEATURES)
    X_np = X.values.astype(np.float32)
    assert X_np.shape[1] == TIMESTEPS * \
        FEATURES, f"Expected {TIMESTEPS * FEATURES} features, got {X_np.shape[1]}"
    X_np = X_np.reshape(-1, TIMESTEPS, FEATURES)

    # normalization per-dataset
    mean = X_np.mean(axis=(0, 1), keepdims=True)
    std = X_np.std(axis=(0, 1), keepdims=True) + 1e-8
    X_np = (X_np - mean) / std

    X_t = torch.tensor(X_np, dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.long)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    return loader, mean, std


def fine_tune_on_member3(base_ckpt_path: str, out_ckpt_path: str, epochs: int = 5, lr: float = 1e-4):
    """
    Load a base checkpoint, fine-tune on Member3 data, and save a new checkpoint.
    """
    print(f"\n=== Fine-tuning {base_ckpt_path} on Member3 ===")

    # Load Member3 data (full dataset as training data)
    train_loader, mean_M3, std_M3 = load_member_dataset(member_id=3)

    # Build model with same architecture
    model = LSTMModel(TIMESTEPS, FEATURES, GLOBAL_NUM_CLASSES).to(DEVICE)

    # Load base weights
    ckpt = torch.load(base_ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(Xb)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += len(Xb)

        avg_loss = running_loss / total
        avg_acc = correct / total
        print(f"[{os.path.basename(base_ckpt_path)}] Epoch {epoch+1}/{epochs} "
              f"Loss={avg_loss:.4f} Acc={avg_acc:.4f}")

    # Save fine-tuned model (also store Member3 normalization)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "timesteps": TIMESTEPS,
            "features": FEATURES,
            "num_classes": GLOBAL_NUM_CLASSES,
            "mean_member3": mean_M3,
            "std_member3": std_M3,
        },
        out_ckpt_path,
    )

    print(f"Saved fine-tuned model to: {out_ckpt_path}")


if __name__ == "__main__":
    # Paths relative to lstm/federated/
    CENTRAL_BASE = os.path.join(
        "..", "centralized", "base_lstm_member2_centralized.pt")
    FED_BASE = "base_lstm_member2_federated.pt"

    CENTRAL_FT = "centralized_finetuned_member3.pt"
    FED_FT = "federated_finetuned_member3.pt"

    # Fine-tune centralized base model on Member3
    if os.path.exists(CENTRAL_BASE):
        fine_tune_on_member3(CENTRAL_BASE, CENTRAL_FT)
    else:
        print(f"WARNING: Centralized base model not found at {CENTRAL_BASE}")

    # Fine-tune federated base model on Member3
    if os.path.exists(FED_BASE):
        fine_tune_on_member3(FED_BASE, FED_FT)
    else:
        print(f"WARNING: Federated base model not found at {FED_BASE}")
