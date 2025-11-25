"""cnn: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
# tqdm import removed
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self, n_input_channels=3, n_classes=6):
        super(Net, self).__init__()
        
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



fds = None  # Cache FederatedDataset



def load_data(partition_id: int, num_partitions: int):
    
    """
    Loads data from local WISDM2 CSV files and partitions it based on subject_id.
    
    Args:
        partition_id (int): The ID of the partition to load (0 to num_partitions-1).
        num_partitions (int): The total number of partitions (clients) to split the subjects into.
        
    Returns:
        train_loader, test_loader
    """
    
    # 1. Load the raw data from CSV files
    # We use header=None because the snippets showed no headers
    try:
        df_x = pd.read_csv('/Users/claudian/Documents/Master_Inteligenta_Artificiala/AIforTDM/AIforTDM-IA-CRM/data/splits/Member1_X.csv', header=None)
        df_y = pd.read_csv('/Users/claudian/Documents/Master_Inteligenta_Artificiala/AIforTDM/AIforTDM-IA-CRM/data/splits/Member1_Y.csv', header=None)
        df_subj = pd.read_csv('/Users/claudian/Documents/Master_Inteligenta_Artificiala/AIforTDM/AIforTDM-IA-CRM/data/splits/Member1_subject_id.csv', header=None)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find one of the required CSV files (Member1_X, Member1_Y, Member1_subject_id). {e}")

    # Ensure the data aligns
    if not (len(df_x) == len(df_y) == len(df_subj)):
        raise ValueError("Input CSV files must have the same number of rows.")

    X_raw = df_x.values.astype(np.float32)
    y = df_y.values.flatten().astype(np.int64)
    subjects = df_subj.values.flatten()
    
    print(f"[-] Raw X shape: {X_raw.shape}")
    
    # Reshape X from (N, 300) to (N, 3, 100)
    # Assumption: Data is flattened as [x1..x100, y1..y100, z1..z100] per row
    # If interleaved, this reshape logic might need adjustment.
    n_samples = X_raw.shape[0]
    n_channels = 3
    n_length = 100
    
    # Reshaping to (N, 3, 100)
    X = X_raw.reshape(n_samples, n_channels, n_length)

    # 2. Identify Unique Subjects
    # We get all unique subject IDs and sort them to ensure deterministic partitioning
    # (i.e., partition 0 always gets the same subjects every time this runs).
    unique_subjects = sorted(df_subj[0].unique())
    total_subjects = len(unique_subjects)

    if num_partitions > total_subjects:
        raise ValueError(f"Cannot create {num_partitions} partitions from only {total_subjects} unique subjects.")

    # 3. Assign Subjects to Partitions
    # We split the list of subjects into 'num_partitions' chunks.
    # np.array_split handles cases where total_subjects isn't perfectly divisible by num_partitions.
    subject_splits = np.array_split(unique_subjects, num_partitions)
    
    # Get the specific subjects assigned to this partition_id
    assigned_subjects = subject_splits[partition_id]
    
    print(f"Partition {partition_id}: Assigned subjects {assigned_subjects}")

    # 4. Filter Data
    # Find the indices in the dataframe where the subject_id matches the assigned subjects
    mask = df_subj[0].isin(assigned_subjects)
    
    # Extract the features (X) and labels (Y) for these subjects
    X_partition = X[mask]
    y_partition = y[mask]

    # 5. Local Train/Test Split
    # We perform a standard 80/20 split on the data specific to this partition.
    # stratify=y_partition ensures the class distribution is preserved in train/test if possible.
    X_train, X_test, y_train, y_test = train_test_split(
        X_partition, 
        y_partition, 
        test_size=0.2, 
        random_state=42
    )

    # Convert to PyTorch Tensors
    tensor_x_train = torch.Tensor(X_train)
    tensor_y_train = torch.LongTensor(y_train)
    tensor_x_test = torch.Tensor(X_test)
    tensor_y_test = torch.LongTensor(y_test)

    # Create Datasets and Loaders
    train_dataset = TensorDataset(tensor_x_train, tensor_y_train)
    test_dataset = TensorDataset(tensor_x_test, tensor_y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    model = net

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = 0
    
    device = torch.device(device)
    print(f"[-] Starting training on {device}...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Standard loop without tqdm
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(trainloader)
        train_losses += epoch_loss
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
    
    train_losses /= epochs
    return train_losses


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    model = net
    model.eval()
    all_preds = []
    all_labels = []
    loss = 0
    correct = 0
    device = torch.device(device)
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy