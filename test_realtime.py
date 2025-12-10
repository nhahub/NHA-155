"""
Simple Real-Time Sign Language Detection Test Script

Usage:
    python test_realtime.py
"""

import cv2
import numpy as np
import torch
import mediapipe as mp
import yaml
import os
import pandas as pd
import re
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display
from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

def load_action_names_from_excel(karsl_root, labels_filename='KARSL-502_Labels.xlsx'):
    """
    Load action names from KArSL Excel file and sanitize them.
    
    Args:
        karsl_root: Path to KArSL-502 dataset root
        labels_filename: Name of the Excel labels file
        
    Returns:
        List of sanitized action names
    """
    labels_path = os.path.join(karsl_root, labels_filename)
    
    if not os.path.exists(labels_path):
        print(f"Warning: Labels file not found at {labels_path}")
        print("Using fallback action names...")
        return None
    
    # Load Excel file
    labels_df = pd.read_excel(labels_path)
    
    # Create mapping from SignID to Arabic sign name (same as utils.py)
    # SignID column: integers 1, 2, 3, ..., 502
    # Sign-Arabic column: Arabic sign names
    sign_mapping = dict(zip(
        labels_df['SignID'].astype(str).str.zfill(4),  # Convert to '0001', '0002', etc.
        labels_df['Sign-Arabic'].astype(str)  # Ensure strings
    ))
    
    # Create ordered list of action names (0001 to 0502)
    action_names = []
    invalid_chars = r'[<>:"/\\|?*]'
    
    for i in range(1, 503):  # 1 to 502
        action_id = str(i).zfill(4)  # '0001', '0002', etc.
        action_name = sign_mapping.get(action_id, f'Unknown-{action_id}')
        
        # Sanitize folder names - replace invalid Windows characters with hyphens
        sanitized_name = re.sub(invalid_chars, '-', action_name).strip()
        action_names.append(sanitized_name)
    
    return action_names


def load_config(config_path='test_config.yaml'):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        print("Creating default config...")
        create_default_config(config_path)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def create_default_config(config_path='test_config.yaml'):
    """Create a default configuration file"""
    default_config = {
        'model': {'path': 'models/best_model.pth'},
        'actions': {'start_index': 0, 'end_index': 118},
        'detection': {'threshold': 0.8, 'sequence_length': 30, 'sentence_length': 5},
        'camera': {'device_id': 0, 'min_detection_confidence': 0.5, 'min_tracking_confidence': 0.5},
        'dataset': {'karsl_root': 'D:/KArSL-502', 'labels_file': 'KARSL-502_Labels.xlsx'}
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"Default config created at: {config_path}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def draw_arabic_text(image, text, position, font_size=32, color=(255, 255, 255)):
    """Draw Arabic text on image using PIL"""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    draw.text(position, bidi_text, font=font, fill=color)
    
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def prob_viz(res, actions, input_frame):
    """Visualize top 5 predictions with probabilities"""
    output_frame = input_frame.copy()
    
    top_indices = np.argsort(res)[-5:][::-1]
    top_probs = res[top_indices]
    top_actions = [actions[i] for i in top_indices]
    
    for num, (action, prob) in enumerate(zip(top_actions, top_probs)):
        color = (int(245 * (1-prob)), int(117 + 138*prob), 16)
        cv2.rectangle(output_frame, (0, 60+num*35), (int(prob*300), 85+num*35), color, -1)
        text_to_display = f"{action}: {prob:.2f}"
        output_frame = draw_arabic_text(output_frame, text_to_display, (5, 60+num*35), 
                                       font_size=20, color=(255, 255, 255))
    
    return output_frame


def load_model(model_path, num_classes, device):
    """Load PyTorch model - supports both simple and full models"""
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if it's a simple model (just state_dict) or full model (with metadata)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Full model from train_pytorch_model_optimized.py
        state_dict = checkpoint['model_state_dict']
        
        # Import the model class
        from train_pytorch_model_optimized import SignLanguageLSTM
        
        # Detect parameters from state dict
        hidden_size = state_dict['lstm.weight_ih_l0'].shape[0] // 4
        
        # Count LSTM layers by checking all layer indices
        lstm_layer_count = 0
        for key in state_dict.keys():
            if 'lstm.weight_ih_l' in key and '_reverse' not in key:
                layer_num = int(key.split('_l')[1].split('.')[0])
                lstm_layer_count = max(lstm_layer_count, layer_num + 1)
        
        print(f"Loading full model: {lstm_layer_count} LSTM layers, hidden_size={hidden_size}, num_classes={num_classes}")
        
        model = SignLanguageLSTM(
            input_size=1662,
            hidden_size=hidden_size,
            num_classes=num_classes,
            num_layers=lstm_layer_count,
            dropout=0.5
        )
        model.load_state_dict(state_dict)
        
    else:
        # Simple model from train_simple.py (just state_dict)
        state_dict = checkpoint
        
        # Import the simple model class
        from train_simple import SimpleLSTM
        
        print(f"Loading simple model: num_classes={num_classes}")
        
        model = SimpleLSTM(
            input_size=1662,
            num_classes=num_classes
        )
        model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    return model


# ============================================================================
# MAIN DETECTION
# ============================================================================

def main():
    # Load configuration
    config = load_config('test_config.yaml')
    
    # Extract config values
    MODEL_PATH = config['model']['path']
    START_ACTION = config['actions']['start_index']
    END_ACTION = config['actions']['end_index']
    THRESHOLD = config['detection']['threshold']
    SEQUENCE_LENGTH = config['detection']['sequence_length']
    SENTENCE_LENGTH = config['detection']['sentence_length']
    CAMERA_ID = config['camera']['device_id']
    MIN_DETECTION_CONF = config['camera']['min_detection_confidence']
    MIN_TRACKING_CONF = config['camera']['min_tracking_confidence']
    
    # Load action names from Excel
    print("Loading action names from dataset...")
    ACTION_NAMES = load_action_names_from_excel(
        config['dataset']['karsl_root'],
        config['dataset']['labels_file']
    )
    
    if ACTION_NAMES is None:
        print("Error: Could not load action names from Excel file")
        print("Please check your dataset path in test_config.yaml")
        return
    
    print(f"Loaded {len(ACTION_NAMES)} action names from dataset")
    
    # Get actions for specified range
    action_indices = list(range(START_ACTION, END_ACTION + 1))
    actions = [ACTION_NAMES[i] for i in action_indices]
    
    # Add null class if configured
    INCLUDE_NULL = config['actions'].get('include_null', False)
    if INCLUDE_NULL:
        action_indices.append(999)
        actions.append('null_no_sign')
    
    print(f"\n{'='*60}")
    print(f"Real-Time Sign Language Detection")
    print(f"{'='*60}")
    print(f"Model: {MODEL_PATH}")
    print(f"Actions: {len(actions)} (indices {START_ACTION} to {END_ACTION})")
    if INCLUDE_NULL:
        print(f"Null class: Enabled (for idle/no sign detection)")
    print(f"Threshold: {THRESHOLD}")
    print(f"\nPress 'q' to quit")
    print(f"{'='*60}\n")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    model = load_model(MODEL_PATH, len(actions), device)
    
    # Detection variables
    sequence = []
    sentence = []
    predictions = []
    confidence_window = []  # Track confidence over time
    smoothing_window = 10  # Frames for majority voting
    
    # MediaPipe setup
    mp_holistic = mp.solutions.holistic
    
    cap = cv2.VideoCapture(CAMERA_ID)
    
    with mp_holistic.Holistic(min_detection_confidence=MIN_DETECTION_CONF, 
                              min_tracking_confidence=MIN_TRACKING_CONF) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            
            # Visual feedback for hand detection
            hand_detected = results.left_hand_landmarks or results.right_hand_landmarks
            pose_detected = results.pose_landmarks
            
            # Show detection status
            status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
            status_text = "Hands: Detected" if hand_detected else "Hands: Not Detected"
            cv2.putText(image, status_text, (10, image.shape[0]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            # Prediction logic - ONLY if hands detected
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-SEQUENCE_LENGTH:]
            
            if len(sequence) == SEQUENCE_LENGTH:
                # Predict
                with torch.no_grad():
                    x = torch.FloatTensor(np.expand_dims(sequence, axis=0)).to(device)
                    output = model(x)
                    res = torch.softmax(output, dim=1).cpu().numpy()[0]
                
                predicted_idx = np.argmax(res)
                predicted_conf = res[predicted_idx]
                predicted_action = actions[predicted_idx]
                
                # Check if prediction is null class
                is_null_prediction = 'null' in predicted_action.lower()
                
                # Track confidence over time for smoothing
                confidence_window.append((predicted_idx, predicted_conf))
                confidence_window = confidence_window[-smoothing_window:]
                
                # Majority voting over smoothing window
                if len(confidence_window) >= smoothing_window:
                    # Get most common prediction in window
                    recent_predictions = [p[0] for p in confidence_window]
                    unique, counts = np.unique(recent_predictions, return_counts=True)
                    majority_idx = unique[np.argmax(counts)]
                    majority_action = actions[majority_idx]
                    
                    # Average confidence for majority prediction
                    majority_confidences = [c for p, c in confidence_window if p == majority_idx]
                    avg_confidence = np.mean(majority_confidences)
                    
                    # Only show if stable and confident
                    if avg_confidence > 0.6:
                        print(f"Detected: {majority_action} ({avg_confidence:.2f})")
                    else:
                        print(f"Uncertain: {majority_action} ({avg_confidence:.2f})")
                    
                    # Add to sentence only if very stable, confident, AND not null class
                    is_null_majority = 'null' in majority_action.lower()
                    if avg_confidence > THRESHOLD and np.max(counts) >= smoothing_window * 0.7 and not is_null_majority:
                        if len(sentence) == 0 or majority_action != sentence[-1]:
                            sentence.append(majority_action)
                else:
                    # Not enough frames for smoothing yet
                    if predicted_conf > 0.6:
                        print(f"Buffering: {predicted_action} ({predicted_conf:.2f})")
                
                if len(sentence) > SENTENCE_LENGTH:
                    sentence = sentence[-SENTENCE_LENGTH:]
                
                # Visualize probabilities
                image = prob_viz(res, actions, image)
            
            # Draw sentence bar
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            
            if len(sentence) > 0:
                sentence_text = ' '.join(sentence)
                image = draw_arabic_text(image, sentence_text, (3, 5), 
                                       font_size=28, color=(255, 255, 255))
            
            cv2.imshow('Sign Language Detection - Press Q to Quit', image)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n\nDetection stopped")
    if len(sentence) > 0:
        print(f"Final sentence: {' '.join(sentence)}")


if __name__ == '__main__':
    main()
