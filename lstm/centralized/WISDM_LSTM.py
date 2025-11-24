import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --------------------------
# 1. LOAD DATA
# --------------------------

X = pd.read_csv("../../splits/Member2_X.csv")
y = pd.read_csv("../../splits/Member2_Y.csv")

print("Loaded shapes:", X.shape, y.shape)

# --------------------------
# 2. LABEL ENCODE
# --------------------------

le = LabelEncoder()
y_enc = le.fit_transform(y.values.ravel())
num_classes = len(le.classes_)
print("Classes:", le.classes_)

# --------------------------
# 3. RESHAPE FOR WISDM2
# --------------------------

# WISDM2 processed = 100 time steps x 3 axes = 300 cols
assert X.shape[1] == 300, f"Expected 300 features, got {X.shape[1]}"

timesteps = 100
features = 3

X_np = X.values.reshape(-1, timesteps, features).astype(np.float32)
print("Reshaped X:", X_np.shape)

# --------------------------
# 4. NORMALIZATION (CRITICAL FOR HAR)
# --------------------------

# normalize per-feature across entire dataset
mean = X_np.mean(axis=(0,1), keepdims=True)
std = X_np.std(axis=(0,1), keepdims=True) + 1e-8
X_np = (X_np - mean) / std

print("Normalized X")

# --------------------------
# 5. TRAIN/TEST SPLIT
# --------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_np, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

# --------------------------
# 6. BUILD LSTM MODEL
# --------------------------

class LSTMNet(nn.Module):
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
        x = x[:, -1, :]  # last timestep
        return self.fc(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMNet(timesteps, features, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print(model)

# --------------------------
# 7. TRAINING LOOP
# --------------------------

epochs = 15

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(Xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(Xb)
        correct += (preds.argmax(dim=1) == yb).sum().item()
        total += len(Xb)

    avg_loss = epoch_loss / total
    avg_acc = correct / total

    print(f"Epoch {epoch+1}/{epochs}  Loss={avg_loss:.4f}  Acc={avg_acc:.4f}")

# --------------------------
# 8. EVALUATION
# --------------------------

model.eval()
correct = 0
total = 0
test_loss = 0

with torch.no_grad():
    for Xb, yb in test_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        preds = model(Xb)
        loss = criterion(preds, yb)

        test_loss += loss.item() * len(Xb)
        correct += (preds.argmax(dim=1) == yb).sum().item()
        total += len(Xb)

test_acc = correct / total
test_loss /= total

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
