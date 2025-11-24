import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# ============================================================
#  GLOBAL CONFIGURATION (SHARED ACROSS ALL CLIENTS)
# ============================================================

GLOBAL_NUM_CLASSES = 6

TIMESTEPS = 100
FEATURES = 3
EXPECTED_FEATURE_COUNT = TIMESTEPS * FEATURES   # = 300

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
#  LSTM MODEL
# ============================================================
class LSTMModel(nn.Module):
    def __init__(self, timesteps, features, num_classes):
        super().__init__()
        self.lstm1 = nn.LSTM(features, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x = x[:, -1, :]   # last time step
        return self.fc(x)


# ============================================================
#  LOAD LOCAL CLIENT DATA
# ============================================================
def load_client_data(cid):

    X = pd.read_csv(f"lstm/federated/clients/X_client_{cid}.csv",
                header=None)
    y = pd.read_csv(f"lstm/federated/clients/y_client_{cid}.csv",
                header=None).iloc[:, 0].astype(int)

    # --- Shape check ---
    assert X.shape[1] == EXPECTED_FEATURE_COUNT, \
        f"Expected {EXPECTED_FEATURE_COUNT} features, got {X.shape[1]}"

    # --- Reshape into (N, 100, 3) ---
    X_np = X.values.reshape(-1, TIMESTEPS, FEATURES).astype(np.float32)

    # --- Normalize ---
    mean = X_np.mean(axis=(0, 1), keepdims=True)
    std = X_np.std(axis=(0, 1), keepdims=True) + 1e-8
    X_np = (X_np - mean) / std

    # --- Train/Test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y.values, test_size=0.2, random_state=42, stratify=y.values
    )

    # --- Convert to Tensors ---
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, X_test, y_train, y_test


# ============================================================
#  FLOWER CLIENT IMPLEMENTATION
# ============================================================
class FlowerClient(fl.client.NumPyClient):

    def __init__(self, cid):
        self.cid = cid

        # Load data
        self.X_train, self.X_test, self.y_train, self.y_test = load_client_data(cid)

        # Build model – using global number of classes
        self.model = LSTMModel(
            TIMESTEPS, FEATURES, GLOBAL_NUM_CLASSES
        ).to(DEVICE)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.train_loader = DataLoader(
            TensorDataset(self.X_train, self.y_train), batch_size=64, shuffle=True
        )
        self.test_loader = DataLoader(
            TensorDataset(self.X_test, self.y_test), batch_size=64, shuffle=False
        )

    # ------------------ Model parameter handling ------------------
    def get_parameters(self, config=None):
        return [v.cpu().numpy() for _, v in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for (key, _), param in zip(state_dict.items(), parameters):
            state_dict[key] = torch.tensor(param)
        self.model.load_state_dict(state_dict)

    # ------------------ Training ------------------
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        local_epochs = int(config.get("local_epochs", 2))
        total_loss, total_correct, total_samples = 0, 0, 0

        for _ in range(local_epochs):
            for Xb, yb in self.train_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)

                self.optimizer.zero_grad()
                preds = self.model(Xb)
                loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * len(Xb)
                total_correct += (preds.argmax(1) == yb).sum().item()
                total_samples += len(Xb)

        return (
            self.get_parameters(),
            total_samples,
            {
                "loss": total_loss / total_samples,
                "accuracy": total_correct / total_samples
            }
        )

    # ------------------ Evaluation ------------------
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        total_loss, total_correct, total_samples = 0, 0, 0

        with torch.no_grad():
            for Xb, yb in self.test_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                preds = self.model(Xb)
                loss = self.criterion(preds, yb)

                total_loss += loss.item() * len(Xb)
                total_correct += (preds.argmax(1) == yb).sum().item()
                total_samples += len(Xb)

        return (
            total_loss / total_samples,
            total_samples,
            {"accuracy": total_correct / total_samples}
        )


# Factory
def client_fn(cid):
    return FlowerClient(cid)
