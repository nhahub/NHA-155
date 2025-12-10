# KArSL-502 Arabic Sign Language Detection

Deep learning system for real-time Arabic Sign Language detection using MediaPipe Holistic keypoints and PyTorch LSTM models. Supports 120+ KArSL (Kuwait Arabic Sign Language) action classes with null class detection for idle states.

## 📋 Features

- **MediaPipe Holistic Integration**: Extracts 1662 features (pose, hands, face landmarks)
- **Flexible Preprocessing Pipeline**: Config-driven preprocessing with incremental dataset building
- **Dual Training Scripts**: 
  - `train_simple.py` - Lightweight BiLSTM (1.9M params) for quick validation
  - `train_pytorch_model_optimized.py` - Advanced 3-layer LSTM (8.5M params) with attention
- **Null Class Support**: Detects "no sign" idle states to reduce false predictions
- **Automatic Class Filtering**: Handles classes with insufficient samples for stratified splitting
- **Real-time Detection**: Webcam-based inference with confidence thresholding and smoothing

## 🏗️ Architecture

### SimpleLSTM (1.9M parameters)
```
Input (30 frames × 1662 features)
    ↓
Bidirectional LSTM (128 units) → BatchNorm → Dropout(0.3)
    ↓
LSTM (64 units) → BatchNorm → Dropout(0.4)
    ↓
Dense (128) → BatchNorm → Dropout(0.5)
    ↓
Dense (64) → Dropout(0.3)
    ↓
Output (num_classes)
```

**Performance**: 92.8% accuracy on 11 classes, 85-92% expected on 120 classes

### Optimized Model (8.5M parameters)
- 3-layer Bidirectional LSTM (256 units)
- Multi-head self-attention (8 heads)
- Layer normalization + residual connections
- Progressive dense layers: 512 → 256 → 128 → output

## 📁 Project Structure

```
Action Detection/
├── utils.py                          # MediaPipe processing utilities
├── preprocess_dataset.py             # Flexible preprocessing with modes
├── preprocess_config.yaml            # Preprocessing configuration
├── train_simple.py                   # Lightweight LSTM training
├── train_pytorch_model_optimized.py  # Advanced training script
├── config.yaml                       # Training configuration
├── record_null_class.py              # Record idle/no-sign sequences
├── add_null_to_existing.py           # Add null class to existing datasets
├── MP_Data_KArSL/                    # Processed keypoint sequences
│   ├── 0/ ... 119/                   # Action class folders
│   └── null_no_sign/                 # Null class sequences
├── preprocessed_data/                # Preprocessed .npz files
│   └── sequences_actions0-119_seq30_with_null_*.npz
├── models/                           # Trained model checkpoints
└── logs_pytorch/                     # Training logs and plots
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create conda environment
conda create -n signlang python=3.10
conda activate signlang

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install mediapipe opencv-python numpy scikit-learn matplotlib seaborn pyyaml tqdm
```

### 2. Data Preprocessing

**Option A: Process from scratch**
```bash
# Configure preprocessing in preprocess_config.yaml
python preprocess_dataset.py
```

**Option B: Add to existing dataset**
```yaml
# preprocess_config.yaml
mode:
  type: 'append'
  existing_file: 'preprocessed_data/sequences_actions0-119_seq30.npz'

actions:
  start_index: 120
  end_index: 150

null_class:
  enabled: true
  folder_name: 'null_no_sign'
  class_index: 999
```

### 3. Record Null Class (Optional but Recommended)

```bash
python record_null_class.py
# Press 'q' to record, 's' to skip, 'ESC' to exit
# Target: 100+ sequences of idle/no-sign states
```

### 4. Training

**Quick validation (Simple LSTM)**
```bash
python train_simple.py
# Trains in ~30-60 minutes on GPU
# Good for testing preprocessing pipeline
```

**Full training (Optimized Model)**
```bash
# Update config.yaml with preprocessed file path
python train_pytorch_model_optimized.py
# Trains in 3-5 hours on GPU
# Best accuracy for production use
```

## ⚙️ Configuration

### Preprocessing Config (`preprocess_config.yaml`)

```yaml
mode:
  type: 'new'  # or 'append'

actions:
  start_index: 0
  end_index: 120

null_class:
  enabled: true
  folder_name: 'null_no_sign'
  class_index: 999

sequence:
  length: 30

output:
  directory: 'preprocessed_data'
```

### Training Config (`config.yaml`)

```yaml
paths:
  preprocessed_file: 'preprocessed_data/sequences_actions0-119_seq30_with_null_*.npz'

model:
  hidden_size: 256
  lstm_layers: 3
  dropout: 0.5

training:
  batch_size: 264
  learning_rate: 0.0005
  num_epochs: 300
  patience: 30
```

## 📊 Dataset

**KArSL-502**: Kuwait Arabic Sign Language dataset with 502 action classes
- ~75,000 video sequences
- MediaPipe Holistic keypoints (1662 features per frame)
- 30 frames per sequence
- Null class for "no sign" detection

**Preprocessed Format** (.npz files):
```python
data = np.load('sequences.npz')
X = data['X']              # (N, 30, 1662) sequences
y = data['y']              # (N,) labels
actions = data['actions']  # Class indices
action_names = data['action_names']  # Class names
```

## 🔧 Key Features

### Automatic Class Filtering
Automatically removes classes with < 10 samples to prevent stratification errors:
```python
# Both training scripts handle this automatically
✓ Filtered dataset: 17,960 sequences, 115 classes
```

### Flexible Preprocessing Modes

**Mode: 'new'** - Create from scratch
```bash
python preprocess_dataset.py  # Creates new .npz file
```

**Mode: 'append'** - Add to existing
```bash
# Combines existing data with new actions
# Detects overlaps and creates smart filenames
```

### Null Class Integration
```python
# In utils.py
X, y, actions, action_names = load_processed_keypoints(
    include_null_class=True,
    null_class_folder='null_no_sign'
)
# Null class assigned to index 999
```

## 📈 Training Results

**Simple LSTM (11 classes)**
- Accuracy: 92.8%
- Training: ~30 minutes
- Parameters: 1.9M

**Expected Performance (120 classes)**
- Simple LSTM: 85-92%
- Optimized Model: 88-95%
- Training: 3-5 hours

## 🎯 Real-time Detection

```python
import cv2
from utils import mediapipe_detection, extract_keypoints

# Load model
model.load_state_dict(torch.load('models/simple_model_best.pth'))

# Webcam loop
cap = cv2.VideoCapture(0)
sequence = []

while True:
    ret, frame = cap.read()
    results = mediapipe_detection(frame, holistic)
    keypoints = extract_keypoints(results)
    
    sequence.append(keypoints)
    sequence = sequence[-30:]  # Keep last 30 frames
    
    if len(sequence) == 30:
        # Predict
        X = torch.FloatTensor([sequence]).to(device)
        output = model(X)
        pred = torch.argmax(output, dim=1).item()
        
        # Display
        cv2.putText(frame, actions[pred], (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Sign Language Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

## 🛠️ Utilities

### Record Null Class
```bash
python record_null_class.py
# Records idle states for robust null class detection
```

### Add Null to Existing Dataset
```bash
python add_null_to_existing.py
# Quickly adds null sequences without full reprocessing
```

### Check Preprocessed Data
```python
data = np.load('preprocessed_data/sequences_*.npz', allow_pickle=True)
print(f"Sequences: {data['X'].shape}")
print(f"Classes: {len(data['actions'])}")
print(f"Metadata: {data['timestamp']}")
```

## 📝 Common Issues

**1. Stratification Error**
```
ValueError: The least populated class in y has only 1 member
```
**Solution**: Automatic filtering removes classes with < 10 samples

**2. Missing Metadata in .npz**
```
KeyError: 'start_action is not a file in the archive'
```
**Solution**: Updated scripts handle both old and new formats gracefully

**3. Null Class Too Small**
```
⚠ Removing class 999 (null_no_sign) - only 50 samples
```
**Solution**: Record 100+ null sequences with `record_null_class.py`

## 📚 Dependencies

- Python 3.10+
- PyTorch 2.7+ (with CUDA 11.8)
- MediaPipe 0.10+
- OpenCV 4.8+
- NumPy, scikit-learn, matplotlib, seaborn, PyYAML, tqdm

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **KArSL-502 Dataset**: Kuwait University
- **MediaPipe**: Google Research
- **PyTorch**: Meta AI Research

## 📧 Contact

For questions or collaborations, please open an issue on GitHub.

---

**Built with ❤️ for the deaf and hard-of-hearing community**
