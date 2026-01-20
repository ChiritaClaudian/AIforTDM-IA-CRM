import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from lstm.federated.client import LSTMModel, TIMESTEPS, FEATURES, GLOBAL_NUM_CLASSES, DEVICE
from lstm.federated.robustness_corruptions import gaussian_noise, channel_dropout, time_mask, time_shift

from pathlib import Path

def load_member_test(member_id: int, ckpt_norm: dict | None):
    # repo_root = two levels up from lstm/federated/robustness_eval.py
    repo_root = Path(__file__).resolve().parents[2]
    splits_dir = repo_root / "splits"

    X = pd.read_csv(splits_dir / f"Member{member_id}_X.csv", header=None).values.astype(np.float32)
    y = pd.read_csv(splits_dir / f"Member{member_id}_Y.csv", header=None).values.ravel().astype(int)

    X = X.reshape(-1, TIMESTEPS, FEATURES)

    if ckpt_norm is not None:
        mean, std = ckpt_norm["mean"], ckpt_norm["std"]
    else:
        mean = X.mean(axis=(0, 1), keepdims=True)
        std  = X.std(axis=(0, 1), keepdims=True) + 1e-8

    X = (X - mean) / (std + 1e-8)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return torch.tensor(X_te, dtype=torch.float32), torch.tensor(y_te, dtype=torch.long)


def load_ckpt_norm(ckpt: dict):
    # centralized baseline saves "mean"/"std"; your fine-tune saves "mean_member3"/"std_member3"
    if "mean" in ckpt and "std" in ckpt:
        return {"mean": ckpt["mean"], "std": ckpt["std"]}
    if "mean_member3" in ckpt and "std_member3" in ckpt:
        return {"mean": ckpt["mean_member3"], "std": ckpt["std_member3"]}
    return None

@torch.no_grad()
def eval_acc(model: nn.Module, X: torch.Tensor, y: torch.Tensor, batch_size: int = 256):
    model.eval()
    correct = 0
    total = 0
    for i in range(0, len(X), batch_size):
        xb = X[i:i+batch_size].to(DEVICE)
        yb = y[i:i+batch_size].to(DEVICE)
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += len(yb)
    return correct / total if total else 0.0

def main():
    import os

    # Evaluate robustness on these checkpoints
    models = [
        ("centralized_member2", os.path.join("lstm", "centralized", "base_lstm_member2_centralized.pt")),
        ("federated_member2",   os.path.join("lstm", "federated",  "base_lstm_member2_federated.pt")),
        ("centralized_ft_m3",   os.path.join("lstm", "federated",  "centralized_finetuned_member3.pt")),
        ("federated_ft_m3",     os.path.join("lstm", "federated",  "federated_finetuned_member3.pt")),
    ]

    # Which member test split(s) to run on
    member_ids = [1, 2, 3]

    results = []

    for model_name, ckpt_path in models:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        ckpt_norm = load_ckpt_norm(ckpt)

        model = LSTMModel(TIMESTEPS, FEATURES, GLOBAL_NUM_CLASSES).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])

        for member_id in member_ids:
            X_te, y_te = load_member_test(member_id, ckpt_norm)

            # Clean
            acc = eval_acc(model, X_te, y_te)
            results.append((model_name, ckpt_path, member_id, "clean", "none", 0, acc))

            # Gaussian noise
            for sigma in [0.0, 0.05, 0.1, 0.2]:
                Xc = gaussian_noise(X_te.to(DEVICE), sigma).cpu()
                acc = eval_acc(model, Xc, y_te)
                results.append((model_name, ckpt_path, member_id, "gaussian", "sigma", sigma, acc))

            # Channel dropout
            for p in [0.0, 0.1, 0.3, 0.5]:
                Xc = channel_dropout(X_te.to(DEVICE), p).cpu()
                acc = eval_acc(model, Xc, y_te)
                results.append((model_name, ckpt_path, member_id, "ch_dropout", "p", p, acc))

            # Time mask
            for L in [0, 5, 10, 20, 40]:
                Xc = time_mask(X_te.to(DEVICE), L).cpu()
                acc = eval_acc(model, Xc, y_te)
                results.append((model_name, ckpt_path, member_id, "time_mask", "L", L, acc))

            # Time shift
            for k in [0, 2, 5, 10, 20]:
                Xc = time_shift(X_te.to(DEVICE), k).cpu()
                acc = eval_acc(model, Xc, y_te)
                results.append((model_name, ckpt_path, member_id, "time_shift", "k", k, acc))

    df = pd.DataFrame(
        results,
        columns=["model", "ckpt_path", "member_id", "corruption", "param", "severity", "accuracy"]
    )
    print(df)
    
    outdir = Path("results")
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "robustness_results.csv"
    df.to_csv(outpath, index=False)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
