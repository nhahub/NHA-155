"""
PyTorch LSTM Training Script for KArSL Sign Language Detection
Uses CUDA GPU acceleration if available
Optimized for KArSL-502: 502 action classes with ~75,000 video sequences
"""

import os
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime
import json
from utils import load_processed_keypoints
from tqdm import tqdm
import yaml

# ============================================================================
# Configuration Loader
# ============================================================================

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "="*80)
    print("Configuration Loaded".center(80))
    print("="*80)
    print(f"Config file: {config_path}")
    print(f"Model: {config['model']['lstm_layers']}-layer LSTM with {config['model']['hidden_size']} hidden units")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Max epochs: {config['training']['num_epochs']}")
    print("="*80)
    
    return config

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_preprocessed_data(preprocessed_file):
    """
    Load preprocessed sequences from .npz file.
    Much faster than processing keypoints every time!
    
    Parameters:
    -----------
    preprocessed_file : str
        Path to .npz file created by preprocess_dataset.py
    
    Returns:
    --------
    tuple : (X, y, actions, action_names, metadata)
    """
    print(f"\nLoading preprocessed data from {preprocessed_file}...")
    
    if not os.path.exists(preprocessed_file):
        raise FileNotFoundError(f"Preprocessed file not found: {preprocessed_file}")
    
    data = np.load(preprocessed_file, allow_pickle=True)
    
    X = data['X']
    y = data['y']
    actions = data['actions']
    action_names = data['action_names']
    
    # Extract metadata (handle both old and new format files)
    metadata = {}
    if 'start_action' in data:
        metadata = {
            'start_action': int(data['start_action']),
            'end_action': int(data['end_action']),
            'sequence_length': int(data['sequence_length']),
            'timestamp': str(data['timestamp']),
            'processing_time': float(data['processing_time'])
        }
    else:
        # Old format file - infer metadata from data
        metadata = {
            'start_action': int(actions[0]),
            'end_action': int(actions[-1]) + 1,
            'sequence_length': X.shape[1],
            'timestamp': 'unknown',
            'processing_time': 0.0
        }
        print("⚠ Warning: Old format file detected (missing metadata)")
    
    file_size = os.path.getsize(preprocessed_file) / 1e6
    
    print(f"✓ Loaded {len(X):,} sequences in {file_size:.1f} MB")
    print(f"  Shape: X={X.shape}, y={y.shape}")
    print(f"  Actions: {metadata['start_action']}-{metadata['end_action']-1} ({len(actions)} classes)")
    print(f"  Sequence length: {metadata['sequence_length']}")
    if metadata['timestamp'] != 'unknown':
        print(f"  Preprocessed on: {metadata['timestamp']}")
    
    return X, y, actions, action_names, metadata

def filter_small_classes(X, y, actions, min_samples=10):
    """
    Filter out classes with too few samples for stratified splitting.
    
    Parameters:
    -----------
    X : array
        Sequences
    y : array
        Labels
    actions : array
        Action names/indices
    min_samples : int
        Minimum samples needed per class (default 10)
    
    Returns:
    --------
    tuple : (X_filtered, y_filtered, actions_filtered)
    """
    unique_classes, class_counts = np.unique(y, return_counts=True)
    
    classes_to_remove = []
    for cls, count in zip(unique_classes, class_counts):
        if count < min_samples:
            classes_to_remove.append(cls)
            print(f"⚠ Removing class {cls} ({actions[cls] if cls < len(actions) else 'unknown'}) - only {count} samples")
    
    if len(classes_to_remove) > 0:
        print(f"\n⚠ Filtering {len(classes_to_remove)} classes with < {min_samples} samples")
        
        # Filter out classes with too few samples
        mask = np.isin(y, classes_to_remove, invert=True)
        X_filtered = X[mask]
        y_filtered = y[mask]
        
        # Remap labels to be contiguous (0, 1, 2, ...)
        unique_remaining = np.unique(y_filtered)
        label_mapping = {old: new for new, old in enumerate(unique_remaining)}
        y_remapped = np.array([label_mapping[label] for label in y_filtered])
        
        # Keep only remaining actions
        actions_filtered = actions[unique_remaining]
        
        print(f"✓ Filtered dataset: {len(X_filtered):,} sequences, {len(actions_filtered)} classes\n")
        
        return X_filtered, y_remapped, actions_filtered
    else:
        print(f"✓ All classes have >= {min_samples} samples\n")
        return X, y, actions

# ============================================================================
# Dataset Class
# ============================================================================

class SignLanguageDataset(Dataset):
    """PyTorch Dataset for sign language sequences"""
    
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# ============================================================================
# Model Architecture (Optimized for 502 classes)
# ============================================================================

class SignLanguageLSTM(nn.Module):
    """Optimized Bidirectional LSTM for KArSL-502 (502 classes, ~75K samples)"""
    
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.5, num_layers=3):
        super(SignLanguageLSTM, self).__init__()
        
        # Multi-layer Bidirectional LSTM with larger capacity
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism for better sequence understanding
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_size * 2)
        self.dropout1 = nn.Dropout(dropout)
        
        # Dense layers with batch normalization
        self.fc1 = nn.Linear(hidden_size * 2, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(dropout * 0.7)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(dropout * 0.5)
        
        # Output layer
        self.fc4 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size*2)
        
        # Self-attention for temporal relationships
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.ln1(attn_out + lstm_out)  # Residual connection
        attn_out = self.dropout1(attn_out)
        
        # Global average pooling over sequence dimension
        pooled = torch.mean(attn_out, dim=1)  # (batch, hidden_size*2)
        
        # Classifier
        out = F.relu(self.fc1(pooled))
        out = self.bn1(out)
        out = self.dropout2(out)
        
        out = F.relu(self.fc2(out))
        out = self.bn2(out)
        out = self.dropout3(out)
        
        out = F.relu(self.fc3(out))
        out = self.bn3(out)
        out = self.dropout4(out)
        
        out = self.fc4(out)
        return out

# ============================================================================
# Training Functions
# ============================================================================

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience=30, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
            self.counter = 0

def train_epoch(model, train_loader, criterion, optimizer, device, grad_clip=None):
    """Train for one epoch with tqdm progress bar"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar for training
    pbar = tqdm(train_loader, desc='Training', leave=False, 
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}')
    
    for sequences, labels in pbar:
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * sequences.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch with tqdm progress bar"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar for validation
    pbar = tqdm(val_loader, desc='Validation', leave=False,
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}')
    
    with torch.no_grad():
        for sequences, labels in pbar:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * sequences.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                scheduler, device, num_epochs, patience, save_path, grad_clip=None):
    """Complete training loop with professional progress tracking"""
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=False)
    
    # History
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    print("\n" + "="*80)
    print("Starting Training".center(80))
    print("="*80)
    
    # Main training loop with epoch progress bar
    epoch_pbar = tqdm(range(num_epochs), desc='Overall Progress', 
                     bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}')
    
    for epoch in epoch_pbar:
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, grad_clip)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'T_Loss': f'{train_loss:.4f}',
            'T_Acc': f'{train_acc*100:.2f}%',
            'V_Loss': f'{val_loss:.4f}',
            'V_Acc': f'{val_acc*100:.2f}%',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        # Print detailed epoch summary every 10 epochs
        if (epoch + 1) % 10 == 0:
            tqdm.write(f'\n[Epoch {epoch+1}/{num_epochs}] '
                      f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | '
                      f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%')
        
        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            tqdm.write(f"\n✓ Early stopping triggered at epoch {epoch+1}")
            model.load_state_dict(early_stopping.best_model)
            break
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, save_path)
    
    print(f"\n✓ Model saved to {save_path}")
    print(f"✓ Best validation loss: {early_stopping.best_loss:.4f}")
    print(f"✓ Best validation accuracy: {max(history['val_acc'])*100:.2f}%")
    
    return history

# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_model(model, test_loader, device, action_names):
    """Evaluate model on test set with progress bar"""
    model.eval()
    all_preds = []
    all_labels = []
    
    # Progress bar for testing
    pbar = tqdm(test_loader, desc='Testing', 
                bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}')
    
    with torch.no_grad():
        for sequences, labels in pbar:
            sequences = sequences.to(device)
            outputs = model(sequences)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    
    print("\n" + "="*80)
    print("Test Results".center(80))
    print("="*80)
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return all_labels, all_preds, accuracy

def plot_training_history(history, save_path, config, dpi=None):
    """Plot training history"""
    if dpi is None:
        dpi = config.get('logging', {}).get('plot_dpi', 300)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot([acc*100 for acc in history['train_acc']], label='Train Acc', linewidth=2)
    axes[1].plot([acc*100 for acc in history['val_acc']], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate plot
    axes[2].plot(history['lr'], linewidth=2, color='green')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Learning Rate', fontsize=12)
    axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"✓ Training history plot saved to {save_path}")

# ============================================================================
# Main Training Script
# ============================================================================

def main():
    # Load configuration
    config = load_config('config.yaml')


    # Set random seeds for reproducibility
    torch.manual_seed(config['data']['random_seed'])
    np.random.seed(config['data']['random_seed'])
    
    # Create directories
    os.makedirs(config['paths']['model_save_path'], exist_ok=True)
    os.makedirs(config['paths']['logs_path'], exist_ok=True)
    
    # Setup device
    use_cuda = config['device']['use_cuda'] and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print("\n" + "="*80)
    print("System Information".center(80))
    print("="*80)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load data - use preprocessed file if available, otherwise process from scratch
    use_preprocessed = config.get('use_preprocessed', False)
    preprocessed_file = config.get('preprocessed_file', None)
    
    if use_preprocessed and preprocessed_file:
        # Load from preprocessed file (FAST!)
        print("\n" + "="*80)
        print("Loading Preprocessed Dataset".center(80))
        print("="*80)
        
        preprocessed_path = os.path.join('preprocessed_data', preprocessed_file)
        X, y, actions, action_names, metadata = load_preprocessed_data(preprocessed_path)
        
        print("✓ Using preprocessed data - much faster than processing keypoints!")
        
    else:
        # Process from scratch (SLOW - only if needed)
        print("\n" + "="*80)
        print("Processing Dataset from Scratch".center(80))
        print("="*80)
        print("⚠ This will take several minutes...")
        print("⚠ Consider running preprocess_dataset.py once and using preprocessed data!")
        print("="*80)
        
        try:
            from utils import load_karsl_dataset_structure
            print("\n✓ Loading dataset structure...")
            dataset_info = load_karsl_dataset_structure(config['paths']['karsl_root'], show_summary=False)
            actions = dataset_info['actions']
            action_names = dataset_info['action_names']
            print(f"✓ Loaded {len(actions)} action classes")
        except Exception as e:
            print(f"\n⚠ Could not load from utils: {e}")
            print("Please update the script with your actions and action_names arrays")
            return
        
        # Load data
        X, y = load_processed_keypoints(actions, action_names, config['paths']['data_path'],
                                        config['data']['sequence_length'], start_action=0, end_action=119)
    
    # Filter out classes with too few samples (prevents stratification errors)
    print("\n" + "="*80)
    print("Filtering Classes".center(80))
    print("="*80)
    X, y, actions = filter_small_classes(X, y, actions, min_samples=10)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['data']['test_size'], random_state=config['data']['random_seed'], stratify=y
    )
    
    # Further split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=config['data']['validation_split'], 
        random_state=config['data']['random_seed'], stratify=y_train
    )
    
    print(f"\n" + "="*80)
    print("Dataset Split".center(80))
    print("="*80)
    print(f"Training samples:   {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Test samples:       {len(X_test):,}")
    print(f"Total samples:      {len(X):,}")
    print(f"Number of classes:  {len(actions)}")
    
    # Create datasets and dataloaders
    train_dataset = SignLanguageDataset(X_train, y_train)
    val_dataset = SignLanguageDataset(X_val, y_val)
    test_dataset = SignLanguageDataset(X_test, y_test)
    
    pin_memory = config['training']['pin_memory'] if use_cuda else False
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], 
                             shuffle=True, num_workers=config['training']['num_workers'], pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], 
                           shuffle=False, num_workers=config['training']['num_workers'], pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], 
                            shuffle=False, num_workers=config['training']['num_workers'], pin_memory=pin_memory)
    
    # Initialize model
    num_classes = len(set(y))  # Use actual number of classes from loaded labels
    model = SignLanguageLSTM(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_classes=num_classes,
        dropout=config['model']['dropout'],
        num_layers=config['model']['lstm_layers']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n" + "="*80)
    print("Model Architecture".center(80))
    print("="*80)
    print(f"Total parameters:      {total_params:,}")
    print(f"Trainable parameters:  {trainable_params:,}")
    print(f"Model size:            {total_params * 4 / 1e6:.2f} MB (float32)")
    print(f"Number of classes:     {num_classes}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], 
                           weight_decay=config['training']['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config['scheduler']['factor'], 
        patience=config['scheduler']['patience'], min_lr=config['scheduler']['min_lr']
    )
    
    # Train model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(config['paths']['model_save_path'], f'karsl502_model_{timestamp}.pth')
    
    history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, config['training']['num_epochs'], config['training']['patience'], 
        model_save_path, config['training']['grad_clip']
    )
    
    # Plot training history
    plot_path = os.path.join(config['paths']['logs_path'], f'training_history_{timestamp}.png')
    plot_training_history(history, plot_path, config)
    
    # Evaluate on test set
    test_labels, test_preds, test_accuracy = evaluate_model(
        model, test_loader, device, action_names
    )
    
    # Save results
    results = {
        'timestamp': timestamp,
        'test_accuracy': float(test_accuracy),
        'num_classes': num_classes,
        'total_samples': len(X),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'num_epochs': len(history['train_loss']),
        'best_val_accuracy': float(max(history['val_acc'])),
        'final_val_loss': float(min(history['val_loss'])),
        'model_parameters': total_params,
        'config': config  # Save entire config
    }
    
    results_path = os.path.join(config['paths']['logs_path'], f'results_{timestamp}.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\n✓ Results saved to {results_path}")
    print("\n" + "="*80)
    print("Training Complete!".center(80))
    print("="*80)
    print(f"Final Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Best Val Accuracy:   {max(history['val_acc'])*100:.2f}%")
    print("="*80)

if __name__ == '__main__':
    main()
