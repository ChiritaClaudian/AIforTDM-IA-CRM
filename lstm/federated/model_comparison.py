import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, TensorDataset

from client import LSTMModel, TIMESTEPS, FEATURES, GLOBAL_NUM_CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_member_dataset(member_id: int):
    """
    Load full Member{member_id} dataset from ../../splits,
    normalize per-member, return DataLoader.
    """
    base_path = os.path.join("..", "..", "splits")
    X_path = os.path.join(base_path, f"Member{member_id}_X.csv")
    Y_path = os.path.join(base_path, f"Member{member_id}_Y.csv")

    X = pd.read_csv(X_path)
    y = pd.read_csv(Y_path)

    y_np = y.values.ravel().astype(int)

    X_np = X.values.astype(np.float32)
    assert X_np.shape[1] == TIMESTEPS * FEATURES, (
        f"Expected {TIMESTEPS * FEATURES} features, got {X_np.shape[1]}"
    )
    X_np = X_np.reshape(-1, TIMESTEPS, FEATURES)

    # per-member normalization
    mean = X_np.mean(axis=(0, 1), keepdims=True)
    std = X_np.std(axis=(0, 1), keepdims=True) + 1e-8
    X_np = (X_np - mean) / std

    X_t = torch.tensor(X_np, dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.long)

    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=False)
    return loader


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            logits = model(Xb)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += len(yb)

    return correct / total if total > 0 else 0.0


def load_model(ckpt_path: str) -> nn.Module:
    model = LSTMModel(TIMESTEPS, FEATURES, GLOBAL_NUM_CLASSES).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


if __name__ == "__main__":
    # 4 models to compare
    models = {
        "central_base": os.path.join("..", "centralized", "base_lstm_member2_centralized.pt"),
        "central_ft_M3": "centralized_finetuned_member3.pt",
        "fed_base": "base_lstm_member2_federated.pt",
        "fed_ft_M3": "federated_finetuned_member3.pt",
    }

    # Load Member1/2/3 datasets
    member_loaders = {}
    for member_id in [1, 2, 3]:
        try:
            loader = load_member_dataset(member_id)
            member_loaders[member_id] = loader
            print(
                f"Loaded Member{member_id} dataset with {len(loader.dataset)} samples.")
        except FileNotFoundError:
            print(
                f"WARNING: Could not find data for Member{member_id}. Skipping.")
            member_loaders[member_id] = None

    # Pretty print header
    print("\nMODEL COMPARISON (clean accuracy, no noise)")
    print("Model               | Member1 | Member2 | Member3")
    print("--------------------+---------+---------+--------")

    for model_name, ckpt_path in models.items():
        if not os.path.exists(ckpt_path):
            print(f"{model_name:20} |   N/A   |   N/A   |   N/A   (missing ckpt)")
            continue

        model = load_model(ckpt_path)

        accs = {}
        for member_id, loader in member_loaders.items():
            if loader is None:
                accs[member_id] = None
            else:
                accs[member_id] = evaluate(model, loader)

        def fmt(a):
            return "  N/A  " if a is None else f"{a*100:6.2f}%"

        print(
            f"{model_name:20} | {fmt(accs[1])} | {fmt(accs[2])} | {fmt(accs[3])}"
        )
