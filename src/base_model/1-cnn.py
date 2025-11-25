import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from dataclasses import dataclass
from tqdm import tqdm 

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
@dataclass
class Args:
    n_instances: int = 17166
    n_dim: int = 3            # Input channels (x, y, z acceleration)
    series_length: int = 100  # Time steps per sample
    n_classes: int = 6        # WISDM typically has 6 classes
    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 5
    test_size: float = 0.2
    random_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # File Paths
    file_x: str = '../data/splits/Member1_X.csv'
    file_y: str = '../data/splits/Member1_Y.csv'
    file_sub: str = '../data/splits/Member1_subject_id.csv'

    @staticmethod
    def from_args():
        parser = argparse.ArgumentParser(description="WISDM 1D CNN Training")
        parser.add_argument('--n_instances', type=int, default=17166)
        parser.add_argument('--n_dim', type=int, default=3)
        parser.add_argument('--series_length', type=int, default=100)
        parser.add_argument('--n_classes', type=int, default=6)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--epochs', type=int, default=20)
        parser.add_argument('--test_size', type=float, default=0.2)
        parser.add_argument('--random_seed', type=int, default=42)
        parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "mps")
        parser.add_argument('--file_x', type=str, default='../data/splits/Member1_X.csv')
        parser.add_argument('--file_y', type=str, default='../data/splits/Member1_Y.csv')
        parser.add_argument('--file_sub', type=str, default='../data/splits/Member1_subject_id.csv')
        
        args = parser.parse_args()
        return Args(**vars(args))

# Standard WISDM Activity Labels
CLASS_LABELS = ['Walking', 'Jogging', 'Stairs', 'Sitting', 'Standing', 'LyingDown']

# ==========================================
# 2. Data Preparation
# ==========================================
def load_data(args):
    """
    Loads WISDM data from specific CSV files for X, Y, and Subject IDs.
    """
    print(f"[-] Loading data from files:\n    X: {args.file_x}\n    Y: {args.file_y}\n    Sub: {args.file_sub}")
    
    try:
        # Load CSVs (assuming no headers based on typical raw exports, or check content)
        # using header=None. If your CSVs have headers, change to header=0
        df_x = pd.read_csv(args.file_x, header=None)
        df_y = pd.read_csv(args.file_y, header=None)
        df_sub = pd.read_csv(args.file_sub, header=None)
        
        # Convert to numpy
        X_raw = df_x.values.astype(np.float32)
        y = df_y.values.flatten().astype(np.int64)
        subjects = df_sub.values.flatten()
        
        print(f"[-] Raw X shape: {X_raw.shape}")
        
        # Reshape X from (N, 300) to (N, 3, 100)
        # Assumption: Data is flattened as [x1..x100, y1..y100, z1..z100] per row
        # If interleaved, this reshape logic might need adjustment.
        n_samples = X_raw.shape[0]
        n_channels = args.n_dim
        n_length = args.series_length
        
        # Reshaping to (N, 3, 100)
        X = X_raw.reshape(n_samples, n_channels, n_length)
        
        print(f"[-] Processed data shapes: X={X.shape}, y={y.shape}, subjects={subjects.shape}")
        return X, y, subjects
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the CSV files are uploaded and paths are correct.")
        raise

def prepare_dataloaders(X, y, subjects, args):
    """
    Splits data based on Subject IDs (Leave-Subjects-Out)
    """
    print("[-] Performing Subject-based Train/Test Split...")
    
    # Get unique subjects
    unique_subjects = np.unique(subjects)
    print(f"[-] Total unique subjects: {len(unique_subjects)}")
    
    # Split subjects into train and test sets
    train_subs, test_subs = train_test_split(
        unique_subjects, 
        test_size=args.test_size, 
        random_state=args.random_seed
    )
    
    print(f"[-] Train Subjects: {len(train_subs)}")
    print(f"[-] Test Subjects: {len(test_subs)}")
    
    # Create boolean masks
    train_mask = np.isin(subjects, train_subs)
    test_mask = np.isin(subjects, test_subs)
    
    # Split data
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    print(f"[-] Train shape: {X_train.shape} | Test shape: {X_test.shape}")

    # Convert to PyTorch Tensors
    tensor_x_train = torch.Tensor(X_train)
    tensor_y_train = torch.LongTensor(y_train)
    tensor_x_test = torch.Tensor(X_test)
    tensor_y_test = torch.LongTensor(y_test)

    # Create Datasets and Loaders
    train_dataset = TensorDataset(tensor_x_train, tensor_y_train)
    test_dataset = TensorDataset(tensor_x_test, tensor_y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader

# ==========================================
# 3. Model Architecture (1D CNN)
# ==========================================
class HAR_CNN(nn.Module):
    def __init__(self, n_input_channels, n_classes):
        super(HAR_CNN, self).__init__()
        
        # Layer 1: Conv -> ReLU -> MaxPool
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=n_input_channels, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2) 
        )
        
        # Layer 2: Conv -> ReLU -> MaxPool
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Layer 3: Conv -> ReLU -> MaxPool
        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Calculate Flatten size dynamically or manually
        # Input 100 -> pool(2) -> 50 -> pool(2) -> 25 -> pool(2) -> 12
        # Final shape: 256 channels * 12 length = 3072
        self.flatten_size = 256 * 12
        
        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, n_classes)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1) # Flatten
        out = self.fc(out)
        return out

# ==========================================
# 4. Training Loop
# ==========================================
def train_model(model, train_loader, test_loader, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    train_losses = []
    val_accuracies = []
    
    device = torch.device(args.device)
    print(f"[-] Starting training on {device}...")
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        # Using tqdm for progress bar
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for i, (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Update progress bar with current loss
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        val_accuracies.append(epoch_acc)
        
        print(f"Val Accuracy: {epoch_acc:.2f}% | Avg Loss: {epoch_loss:.4f}")
        
    return train_losses, val_accuracies

# ==========================================
# 5. Evaluation
# ==========================================
def evaluate_model(model, test_loader, args):
    model.eval()
    all_preds = []
    all_labels = []
    
    device = torch.device(args.device)
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Metrics
    print("\n" + "="*30)
    print("FINAL EVALUATION REPORT")
    print("="*30)
    print(classification_report(all_labels, all_preds, target_names=CLASS_LABELS[:args.n_classes]))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    return cm

# ==========================================
# 6. Main Execution
# ==========================================
if __name__ == '__main__':
    # 0. Parse Args
    args = Args.from_args()
    print(f"[-] Configuration: {args}")

    # 1. Load Data
    try:
        X, y, subjects = load_data(args)
        
        # 2. Prepare Loaders with Subject Split
        train_loader, test_loader = prepare_dataloaders(X, y, subjects, args)
        
        # 3. Initialize Model
        device = torch.device(args.device)
        model = HAR_CNN(n_input_channels=args.n_dim, n_classes=args.n_classes)
        model.to(device)
        
        # 4. Train
        train_losses, val_accs = train_model(model, train_loader, test_loader, args)
        
        # 5. Evaluate
        cm = evaluate_model(model, test_loader, args)
        
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()