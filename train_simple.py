"""
Simple LSTM Training Script - For Quick Testing with Small Datasets
Lightweight model for testing 5-10 actions quickly
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ============================================================================
# Simple Dataset Class
# ============================================================================

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================================
# Simple LSTM Model (Lightweight)
# ============================================================================

class SimpleLSTM(nn.Module):
    """
    Simple LSTM matching your working Keras architecture:
    - Bidirectional LSTM (128 units)
    - LSTM (64 units)  
    - Dense layers with regularization and dropout
    """
    
    def __init__(self, input_size=1662, num_classes=5):
        super(SimpleLSTM, self).__init__()
        
        # First Bidirectional LSTM layer (128 units)
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.bn1 = nn.BatchNorm1d(256)  # 128*2 for bidirectional
        self.dropout1 = nn.Dropout(0.3)
        
        # Second LSTM layer (64 units)
        self.lstm2 = nn.LSTM(
            input_size=256,  # Input from bidirectional LSTM
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.4)
        
        # Dense layers
        self.fc1 = nn.Linear(64, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(128, 64)
        self.dropout4 = nn.Dropout(0.3)
        
        # Output layer
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # First Bidirectional LSTM
        lstm1_out, _ = self.lstm1(x)  # (batch, seq, 256)
        
        # BatchNorm on sequence outputs (need to permute for BatchNorm1d)
        batch_size, seq_len, features = lstm1_out.shape
        lstm1_out = lstm1_out.permute(0, 2, 1)  # (batch, features, seq)
        lstm1_out = self.bn1(lstm1_out)
        lstm1_out = lstm1_out.permute(0, 2, 1)  # (batch, seq, features)
        lstm1_out = self.dropout1(lstm1_out)
        
        # Second LSTM
        lstm2_out, (h_n, c_n) = self.lstm2(lstm1_out)  # (batch, seq, 64)
        
        # Take last output
        last_output = lstm2_out[:, -1, :]  # (batch, 64)
        last_output = self.bn2(last_output)
        last_output = self.dropout2(last_output)
        
        # Dense layers
        out = torch.relu(self.fc1(last_output))
        out = self.bn3(out)
        out = self.dropout3(out)
        
        out = torch.relu(self.fc2(out))
        out = self.dropout4(out)
        
        out = self.fc3(out)
        
        return out

# ============================================================================
# Training Functions
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    
    return total_loss / total, correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    return total_loss / total, correct / total

# ============================================================================
# Main Training
# ============================================================================

def main():
    print("\n" + "="*70)
    print("Simple LSTM Training - Quick Test".center(70))
    print("="*70)
    
    # Load preprocessed data
    # UPDATE THIS PATH after preprocessing completes
    data_file = 'preprocessed_data/sequences_actions0-119_seq30_with_null_20251210_023455.npz'
    
    print(f"\nLoading data from: {data_file}")
    data = np.load(data_file, allow_pickle=True)
    
    X = data['X']
    y = data['y']
    actions = data['actions']
    
    # Check for classes with insufficient samples for stratified split
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_samples_needed = 10  # Minimum for stratified split (70/30 then 50/50 needs ~7-8 samples)
    
    classes_to_remove = []
    for cls, count in zip(unique_classes, class_counts):
        if count < min_samples_needed:
            classes_to_remove.append(cls)
            print(f"\n⚠ Class {cls} ({actions[cls]}) has only {count} samples - removing")
    
    if len(classes_to_remove) > 0:
        print(f"\n⚠ Removing {len(classes_to_remove)} classes with < {min_samples_needed} samples")
        
        # Filter out classes with too few samples
        mask = np.isin(y, classes_to_remove, invert=True)
        X = X[mask]
        y_filtered = y[mask]
        
        # Remap labels to be contiguous (0, 1, 2, ...)
        unique_remaining = np.unique(y_filtered)
        label_mapping = {old: new for new, old in enumerate(unique_remaining)}
        y = np.array([label_mapping[label] for label in y_filtered])
        
        # Keep only remaining actions
        actions = actions[unique_remaining]
        
        print(f"  ✓ Filtered dataset: {len(X)} sequences, {len(actions)} classes")
    
    print(f"\n✓ Final dataset: {len(X)} sequences, {len(actions)} classes")
    print(f"  Shape: X={X.shape}, y={y.shape}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # Create dataloaders
    train_dataset = SimpleDataset(X_train, y_train)
    val_dataset = SimpleDataset(X_val, y_val)
    test_dataset = SimpleDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create model (matching your working Keras architecture)
    num_classes = len(np.unique(y))
    model = SimpleLSTM(
        input_size=1662,
        num_classes=num_classes
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params:,} parameters ({total_params*4/1e6:.1f} MB)")
    print(f"Architecture: BiLSTM(128) → LSTM(64) → Dense(128) → Dense(64) → Output({num_classes})")
    print(f"Classes: {num_classes}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    num_epochs = 300
    best_val_acc = 0
    patience = 30  # Increased for longer training
    patience_counter = 0
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print("\n" + "="*70)
    print("Training Started".center(70))
    print("="*70)
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1:02d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}%")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'models/simple_model_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n✓ Early stopping at epoch {epoch+1}")
                break
    
    # Load best model and test
    model.load_state_dict(torch.load('models/simple_model_best.pth'))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    
    print("\n" + "="*70)
    print("Results".center(70))
    print("="*70)
    print(f"Best Val Accuracy:  {best_val_acc*100:.2f}%")
    print(f"Test Accuracy:      {test_acc*100:.2f}%")
    print("="*70)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot([acc*100 for acc in history['train_acc']], label='Train')
    ax2.plot([acc*100 for acc in history['val_acc']], label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('logs_pytorch/simple_training.png', dpi=150)
    print(f"\n✓ Plot saved to logs_pytorch/simple_training.png")
    
    # Interpretation
    print("\n" + "="*70)
    print("Interpretation".center(70))
    print("="*70)
    if test_acc > 0.8:
        print("✓ EXCELLENT: Model learned well! Your approach works.")
        print("  → Can proceed with full 119 classes")
    elif test_acc > 0.5:
        print("✓ GOOD: Model is learning reasonably well.")
        print("  → Should work with more data (119 classes)")
    elif test_acc > 0.3:
        print("⚠ MODERATE: Model learning but struggling.")
        print("  → May need more sequences per class or tuning")
    else:
        print("✗ POOR: Model not learning effectively.")
        print("  → Check data quality or try different features")
    
    print(f"\nRandom chance accuracy: {100/num_classes:.1f}%")
    print(f"Your model accuracy:    {test_acc*100:.1f}%")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
