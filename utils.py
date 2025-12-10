"""
utils.py - MediaPipe Sign Language Processing Utilities

This module provides utility functions for processing the KArSL-502 Arabic Sign Language dataset.
It includes tools for MediaPipe Holistic keypoint extraction, video processing, dataset exploration,
and output folder management.

Main Components:
- MediaPipe Detection & Visualization: Core functions for pose/hand/face keypoint extraction
- Dataset Loading: Functions to load and explore the KArSL-502 dataset structure
- Video Processing: Functions to process videos and extract keypoint sequences
- Folder Management: Utilities to create and manage output folders for processed data
- Testing Tools: Webcam and video file testing utilities for MediaPipe

Dataset: KArSL-502 
"""

import os
import re
import cv2
import numpy as np
import mediapipe as mp
import seaborn as sns


# ============================================================================
# MEDIAPIPE UTILITIES  
# ============================================================================

# MediaPipe solutions (initialized at module level for convenience)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


# ============================================================================
# MEDIAPIPE DETECTION FUNCTIONS
# ============================================================================


def mediapipe_detection(image, model):
    """
    Process an image frame using MediaPipe Holistic model.
    
    Converts image to RGB, processes it with MediaPipe, and converts back to BGR.
    This function handles the color space conversions required by MediaPipe.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image in BGR format (OpenCV default)
    model : mediapipe.solutions.holistic.Holistic
        MediaPipe Holistic model instance
    
    Returns:
    --------
    tuple : (processed_image, results)
        - processed_image: Image in BGR format with writeable flag restored
        - results: MediaPipe detection results containing pose, hands, and face landmarks
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                   # Image is no longer writeable
    results = model.process(image)                  # Make prediction
    image.flags.writeable = True                    # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR CONVERSION RGB 2 BGR
    return image, results


def extract_keypoints(results):
    """
    Extract and flatten keypoints from MediaPipe Holistic results.
    
    Extracts pose (33 landmarks × 4 = 132 values), face (468 landmarks × 3 = 1404 values),
    left hand (21 landmarks × 3 = 63 values), and right hand (21 landmarks × 3 = 63 values)
    for a total of 1662 values per frame.
    
    Parameters:
    -----------
    results : mediapipe.solutions.holistic.Holistic results
        MediaPipe detection results containing landmark data
    
    Returns:
    --------
    numpy.ndarray : Flattened array of shape (1662,) containing all keypoints
        - pose: 132 values (x, y, z, visibility for 33 landmarks)
        - face: 1404 values (x, y, z for 468 landmarks)
        - left_hand: 63 values (x, y, z for 21 landmarks)
        - right_hand: 63 values (x, y, z for 21 landmarks)
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def draw_landmarks(image, results):
    """
    Draw basic landmarks on image without styling.
    
    Draws face mesh contours, pose skeleton, and hand connections using default MediaPipe styling.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image to draw landmarks on (modified in-place)
    results : mediapipe.solutions.holistic.Holistic results
        MediaPipe detection results containing landmark data
    """
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_face_mesh.FACEMESH_CONTOURS
        )
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
        )


def draw_styled_landmarks(image, results):
    """
    Draw styled landmarks on image with custom colors and thickness.
    
    Draws face mesh, pose skeleton, and hand connections with custom styling:
    - Face: Green tones (80,110,10) and (80,256,121)
    - Pose: Dark red/purple tones (80,22,10) and (80,44,121)
    - Left hand: Purple/pink tones (121,22,76) and (121,44,250)
    - Right hand: Orange tones (245,117,66) and (245,66,230)
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image to draw landmarks on (modified in-place)
    results : mediapipe.solutions.holistic.Holistic results
        MediaPipe detection results containing landmark data
    """
    # Draw face mesh contours
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        )
    # Draw pose connections
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        )
    # Draw left hand connections
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        )
    # Draw right hand connections
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )


# ============================================================================
# TESTING UTILITIES
# ============================================================================

def test_webcam_mediapipe(camera_index=0, min_detection_confidence=0.5, 
                          min_tracking_confidence=0.5, window_name='OpenCV Feed',
                          show_results=False):
    """
    Test MediaPipe Holistic detection on live webcam feed.
    
    This function opens the webcam and continuously processes frames with MediaPipe Holistic,
    displaying the detected landmarks (face, pose, hands) in real-time.
    
    Parameters:
    -----------
    camera_index : int
        Camera device index (default: 0 for default webcam)
    min_detection_confidence : float
        Minimum confidence for MediaPipe detection (default: 0.5)
    min_tracking_confidence : float
        Minimum confidence for MediaPipe tracking (default: 0.5)
    window_name : str
        Name of the display window (default: 'OpenCV Feed')
    show_results : bool
        Whether to print MediaPipe results to console (default: False)
    
    Returns:
    --------
    None
    
    Controls:
    ---------
    - Press 'q' to quit
    
    Example:
    --------
    >>> test_webcam_mediapipe()
    >>> test_webcam_mediapipe(camera_index=1, show_results=True)
    """
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return
    
    print(f"Starting webcam test (Camera {camera_index})")
    print(f"Press 'q' to quit")
    print("-" * 50)
    
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=min_detection_confidence, 
                              min_tracking_confidence=min_tracking_confidence) as holistic:
        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()
            
            if not ret:
                print("Warning: Failed to read frame from camera")
                break

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            
            if show_results:
                print(results)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)

            # Show to screen
            cv2.imshow(window_name, image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Webcam test complete!")

def test_mediapipe_on_video(base_path, action_id='0071', display_scale=0.5):
    """
    Test MediaPipe detection on a sample video from KArSL dataset.
    
    Parameters:
    -----------
    base_path : str
        Path to the base directory containing action folders
    action_id : str
        Action folder ID to test (default: '0071')
    display_scale : float
        Scale factor for display window (default: 0.5 for 50% size)
    """
    import glob
    import os
    
    # Get action folder
    test_action_path = os.path.join(base_path, action_id)
    
    # Get first video file
    video_files = glob.glob(os.path.join(test_action_path, '*.mp4')) + \
                  glob.glob(os.path.join(test_action_path, '*.avi')) + \
                  glob.glob(os.path.join(test_action_path, '*.mov'))
    
    if not video_files:
        print(f"No video files found in {test_action_path}")
        print("Make sure the base_path and action_id are correct!")
        return
    
    test_video_path = video_files[0]
    print(f"Testing with video: {test_video_path}")
    print(f"Action folder: {action_id}")
    
    # Open video
    cap = cv2.VideoCapture(test_video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file!")
        return
    
    print("Video opened successfully!")
    print(f"Total frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()
            
            if not ret:
                print("End of video or cannot read frame")
                break

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # Resize for display
            display_height, display_width = image.shape[:2]
            display_image = cv2.resize(image, 
                                      (int(display_width * display_scale), 
                                       int(display_height * display_scale)))
            
            # Add text overlay with info
            cv2.putText(display_image, f'Action: {action_id}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display_image, 'Press Q to quit', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # Show to screen
            cv2.imshow('KArSL Video Test - MediaPipe Detection', display_image)

            # Break gracefully (press 'q' to quit)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
    cap.release()
    cv2.destroyAllWindows()
    print("✓ Test complete!")


# ============================================================================
# DATASET LOADING FUNCTIONS
# ============================================================================
    """
    Extract and flatten keypoints from MediaPipe Holistic results.
    
    Extracts pose (33 landmarks × 4 = 132 values), face (468 landmarks × 3 = 1404 values),
    left hand (21 landmarks × 3 = 63 values), and right hand (21 landmarks × 3 = 63 values)
    for a total of 1662 values per frame.
    
    Parameters:
    -----------
    results : mediapipe.solutions.holistic.Holistic results
        MediaPipe detection results containing landmark data
    
    Returns:
    --------
    numpy.ndarray : Flattened array of shape (1662,) containing all keypoints
        - pose: 132 values (x, y, z, visibility for 33 landmarks)
        - face: 1404 values (x, y, z for 468 landmarks)
        - left_hand: 63 values (x, y, z for 21 landmarks)
        - right_hand: 63 values (x, y, z for 21 landmarks)
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================



def load_karsl_dataset_structure(karsl_root, output_path='MP_Data_KArSL', sequence_length=30, show_summary=True):
    """
    Load and organize the KArSL-502 dataset structure.
    
    This function scans the KArSL-502 dataset directory to build mappings of actions
    to their video file locations across different persons (01, 02, 03) and data splits
    (train, test). It reads the Excel labels file to map action IDs to Arabic sign names.
    
    Parameters:
    -----------
    karsl_root : str
        Path to the KArSL-502 dataset root directory
    output_path : str
        Path where processed keypoints will be saved (default: 'MP_Data_KArSL')
    sequence_length : int
        Number of frames to extract from each video (default: 30)
    show_summary : bool
        Whether to print dataset summary (default: True)
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'actions': numpy array of action IDs (e.g., ['0001', '0002', ...])
        - 'action_names': numpy array of Arabic sign names
        - 'sign_mapping': dict mapping action IDs to Arabic names
        - 'video_paths_by_action': dict mapping action IDs to list of folder paths
        - 'data_path': path where processed data will be saved
        - 'sequence_length': number of frames per sequence
        - 'labels_df': pandas DataFrame with full label data
    """
    import pandas as pd
    
    # Setup paths
    labels_path = os.path.join(karsl_root, 'KARSL-502_Labels.xlsx')
    data_path = os.path.join(output_path)
    
    # Load labels from Excel file
    labels_df = pd.read_excel(labels_path)
    
    if show_summary:
        print("Labels DataFrame columns:", labels_df.columns.tolist())
        print("\nFirst few rows:")
        print(labels_df.head())
        print("\nSample rows with Arabic text:")
        print(labels_df[70:75])
    
    # Create mapping from SignID (formatted as 4-digit string) to Arabic sign name
    # SignID column contains integers (1, 2, 3, ..., 502)
    # Sign-Arabic column contains the Arabic sign names
    sign_mapping = dict(zip(
        labels_df['SignID'].astype(str).str.zfill(4),  # Convert SignID to 4-digit strings: 0001, 0002, ...
        labels_df['Sign-Arabic']  # Use Arabic sign names
    ))
    
    if show_summary:
        print(f"\nCreated mapping for {len(sign_mapping)} signs")
        print(f"\nSample mapping (numbers):")
        for k, v in list(sign_mapping.items())[:5]:
            print(f"  {k} -> {v}")
        print(f"\nSample mapping (Arabic signs):")
        for k, v in list(sign_mapping.items())[70:75]:
            print(f"  {k} -> {v}")
    
    # Get all person folders (01, 02, 03) and dataset splits (train, test)
    person_folders = ['01', '02', '03']
    dataset_splits = ['train', 'test']
    
    # Collect all action folders and their paths from all persons and splits
    all_action_folders = set()
    video_paths_by_action = {}  # Maps action_id (4-digit folder name) to list of paths containing videos
    
    for person in person_folders:
        for split in dataset_splits:
            # Check all possible action range folders (e.g., 0001-0100, 0071-0170, etc.)
            person_split_path = os.path.join(karsl_root, person, split)
            
            if os.path.exists(person_split_path):
                # Get all subdirectories in this path (these are action range folders)
                for range_folder in os.listdir(person_split_path):
                    range_path = os.path.join(person_split_path, range_folder)
                    
                    if os.path.isdir(range_path):
                        # Get all action folders within this range
                        for action_folder in os.listdir(range_path):
                            action_path = os.path.join(range_path, action_folder)
                            
                            if os.path.isdir(action_path):
                                all_action_folders.add(action_folder)
                                
                                # Store path for this action
                                if action_folder not in video_paths_by_action:
                                    video_paths_by_action[action_folder] = []
                                video_paths_by_action[action_folder].append(action_path)
    
    # Convert to sorted array
    actions = np.array(sorted(list(all_action_folders)))  # This contains 4-digit folder IDs like '0071', '0162', etc.
    action_names = np.array([sign_mapping.get(folder, f'Unknown-{folder}') for folder in actions])  # Map to Arabic names
    
    if show_summary:
        print(f"\n{'='*70}")
        print(f"DATASET SUMMARY")
        print(f"{'='*70}")
        print(f"Found {len(actions)} unique action classes across all persons and splits")
        print(f"\nFirst 10 actions:")
        for i in range(min(10, len(actions))):
            print(f"  {actions[i]} -> {action_names[i]}")
        
        # Show statistics about the dataset structure
        total_action_folders = sum(len(paths) for paths in video_paths_by_action.values())
        print(f"\n{'='*70}")
        print(f"FOLDER ORGANIZATION")
        print(f"{'='*70}")
        print(f"Total action folders across all persons/splits: {total_action_folders}")
        print(f"  This means: {total_action_folders} physical folders containing videos")
        print(f"  Each action appears in: {total_action_folders / len(actions):.1f} folders on average")
        print(f"  (because each action is recorded by 3 persons × 2 splits = 6 folders)")
        print(f"\nExample - Action '{actions[0]}' ({action_names[0]}):")
        print(f"  Found in {len(video_paths_by_action[actions[0]])} locations:")
        for path in video_paths_by_action[actions[0]]:
            print(f"    - {path}")
    
    # Return all important variables as a dictionary
    return {
        'actions': actions,
        'action_names': action_names,
        'sign_mapping': sign_mapping,
        'video_paths_by_action': video_paths_by_action,
        'data_path': data_path,
        'sequence_length': sequence_length,
        'labels_df': labels_df
    }


def explore_action_folder(actions, action_names, video_paths_by_action, action_index=70, 
                          show_comparison=True, num_comparison_actions=10):
    """
    Explore and display information about a specific action folder from the dataset.
    
    This function helps you inspect the dataset structure by showing:
    - Action ID and Arabic name for a selected action
    - Path where the action videos are stored
    - Number of video files in that action folder
    - Sample filenames
    - Optional comparison with other actions (numbers vs Arabic signs)
    
    Parameters:
    -----------
    actions : numpy.ndarray
        Array of action IDs (e.g., ['0001', '0002', ...])
    action_names : numpy.ndarray
        Array of action names in Arabic
    video_paths_by_action : dict
        Dictionary mapping action IDs to list of folder paths
    action_index : int
        Index of the action to explore (default: 70 for Arabic sign)
    show_comparison : bool
        Whether to show comparison between number and Arabic signs (default: True)
    num_comparison_actions : int
        Number of actions to show in comparison (default: 10)
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'action_id': The action ID (e.g., '0071')
        - 'action_name': The Arabic name of the action
        - 'path': Path to the action folder
        - 'num_files': Number of video files found
        - 'sample_files': List of sample filenames (first 5)
    """
    # Get action information
    sample_action = actions[action_index]
    sample_name = action_names[action_index]
    
    # Use the first path where this action exists
    sample_path = video_paths_by_action[sample_action][0]
    
    print(f"Exploring action ID: {sample_action}")
    print(f"Action name: {sample_name}")
    print(f"Path: {sample_path}")
    
    # List files in the sample action folder
    num_files = 0
    sample_files = []
    
    if os.path.exists(sample_path):
        files = os.listdir(sample_path)
        num_files = len(files)
        sample_files = files[:5]
        print(f"Number of files: {num_files}")
        print(f"Sample files: {sample_files}")
    else:
        print("Path does not exist!")
    
    # Show comparison if requested
    if show_comparison:
        print("\n" + "="*70)
        print(f"For comparison, here are the first {num_comparison_actions} actions (number signs):")
        for i in range(min(num_comparison_actions, len(actions))):
            print(f"  {actions[i]} -> {action_names[i]}")
        
        print(f"\nAnd here are actions {action_index}-{action_index+5} (Arabic word signs):")
        for i in range(action_index, min(action_index + 5, len(actions))):
            print(f"  {actions[i]} -> {action_names[i]}")
    
    # Return summary dictionary
    return {
        'action_id': sample_action,
        'action_name': sample_name,
        'path': sample_path,
        'num_files': num_files,
        'sample_files': sample_files
    }


def count_videos_by_action(actions, video_paths_by_action, show_sample=True, sample_size=10):
    """
    Count video files for each action across all persons and splits.
    
    This function scans all video folders for each action and counts the total
    number of video files (.mp4, .avi, .mov) across all locations (persons and splits).
    
    Parameters:
    -----------
    actions : numpy.ndarray
        Array of action IDs (e.g., ['0001', '0002', ...])
    video_paths_by_action : dict
        Dictionary mapping action IDs to list of folder paths containing videos
    show_sample : bool
        Whether to show sample counts for individual actions (default: True)
    sample_size : int
        Number of sample actions to display (default: 10)
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'total_videos': Total number of video files across all actions
        - 'action_video_counts': Dict mapping action_id -> video count
        - 'average_per_action': Average number of videos per action
        - 'min_count': Minimum video count for any action
        - 'max_count': Maximum video count for any action
    """
    import glob
    
    total_videos = 0
    action_video_counts = {}  # Maps action_id -> total video count across all locations
    
    for action in actions:
        # Get all paths where this action exists (across persons 01, 02, 03 and train/test)
        action_paths = video_paths_by_action.get(action, [])
        
        action_total = 0
        for action_folder in action_paths:
            # Get all video files in this location
            video_files = glob.glob(os.path.join(action_folder, '*.mp4')) + \
                          glob.glob(os.path.join(action_folder, '*.avi')) + \
                          glob.glob(os.path.join(action_folder, '*.mov'))
            action_total += len(video_files)
        
        action_video_counts[action] = action_total
        total_videos += action_total
    
    # Calculate statistics
    counts_list = list(action_video_counts.values())
    avg_per_action = total_videos / len(actions) if len(actions) > 0 else 0
    min_count = min(counts_list) if counts_list else 0
    max_count = max(counts_list) if counts_list else 0
    
    # Display summary
    print("Video Count Summary:")
    print("="*70)
    print(f"Total video files found: {total_videos}")
    print(f"Total actions: {len(actions)}")
    print(f"Average videos per action: {avg_per_action:.1f}")
    print(f"Min videos in any action: {min_count}")
    print(f"Max videos in any action: {max_count}")
    
    # Show sample counts
    if show_sample and sample_size > 0:
        print(f"\nSample action video counts (first {sample_size}):")
        for i, action in enumerate(actions[:sample_size]):
            count = action_video_counts[action]
            print(f"  {action}: {count} videos")
    
    return {
        'total_videos': total_videos,
        'action_video_counts': action_video_counts,
        'average_per_action': avg_per_action,
        'min_count': min_count,
        'max_count': max_count
    }


# ============================================================================
# FOLDER MANAGEMENT FUNCTIONS
# ============================================================================


def create_output_folders(actions, action_names, data_path, show_progress=True):
    """
    Create output folders for processed keypoints.
    
    This function creates a folder structure in the output directory (MP_Data_KArSL)
    with one folder per action, named using the Arabic sign names. These folders will
    store the extracted keypoint sequences (.npy files) during video processing.
    
    Invalid Windows filename characters (< > : " / \ | ? *) are automatically replaced with hyphens (-).
    
    Parameters:
    -----------
    actions : numpy.ndarray
        Array of action IDs (e.g., ['0001', '0002', ...])
    action_names : numpy.ndarray
        Array of action names in Arabic corresponding to each action ID
    data_path : str
        Path where the output folders should be created
    show_progress : bool
        Whether to print progress for each folder created (default: True)
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'total_folders': Total number of folders created/verified
        - 'successful': Number of successfully created folders
        - 'failed': Number of folders that failed to create
        - 'failed_actions': List of action names that failed
        - 'renamed_actions': Dict mapping original names to sanitized names
    """
    import re
    
    total_folders = len(actions)
    successful = 0
    failed = 0
    failed_actions = []
    renamed_actions = {}
    
    print(f"Creating output folders in: {data_path}")
    print(f"Total folders to create: {total_folders}")
    print("=" * 70)
    
    for action, action_name in zip(actions, action_names):
        # Sanitize folder name by replacing invalid Windows characters with hyphen
        invalid_chars = r'[<>:"/\\|?*]'
        sanitized_name = re.sub(invalid_chars, '-', action_name).strip()
        
        # Track if name was changed
        if sanitized_name != action_name:
            renamed_actions[action_name] = sanitized_name
            if show_progress:
                print(f"Sanitized name: '{action_name}' -> '{sanitized_name}'")
        
        action_path = os.path.join(data_path, sanitized_name)
        try:
            os.makedirs(action_path, exist_ok=True)
            successful += 1
            if show_progress:
                print(f"Created/verified folder: {action} -> {sanitized_name}")
        except Exception as e:
            failed += 1
            failed_actions.append(action_name)
            print(f"Error creating folder {sanitized_name}: {e}")
    
    print("\n" + "=" * 70)
    print("FOLDER CREATION SUMMARY:")
    print("=" * 70)
    print(f"Total folders: {total_folders}")
    print(f"Successfully created/verified: {successful}")
    print(f"Failed: {failed}")
    
    if renamed_actions:
        print(f"\nSanitized {len(renamed_actions)} folder names (replaced invalid characters with '-'):")
        for original, sanitized in list(renamed_actions.items())[:5]:
            print(f"  '{original}' -> '{sanitized}'")
        if len(renamed_actions) > 5:
            print(f"  ... and {len(renamed_actions) - 5} more")
    
    if failed > 0:
        print(f"\nFailed actions: {failed_actions}")
    
    return {
        'total_folders': total_folders,
        'successful': successful,
        'failed': failed,
        'failed_actions': failed_actions,
        'renamed_actions': renamed_actions
    }


# ============================================================================
# VIDEO PROCESSING FUNCTIONS
# ============================================================================


def process_karsl_videos_extract_keypoints(actions, action_names, video_paths_by_action, 
                                           data_path, sequence_length=30, 
                                           min_detection_confidence=0.5, 
                                           min_tracking_confidence=0.5,
                                           sanitize_folder_names=True):
    """
    Process KArSL-502 dataset videos and extract keypoints using MediaPipe Holistic.
    
    This function:
    1. Processes videos from all persons (01, 02, 03) and splits (train, test)
    2. Extracts keypoints from each video using MediaPipe Holistic
    3. Saves keypoints as .npy files organized by action and sequence number
    4. Handles videos of varying lengths by sampling frames evenly
    
    Parameters:
    -----------
    actions : numpy.ndarray
        Array of action IDs (e.g., ['0001', '0002', ...])
    action_names : numpy.ndarray
        Array of action names in Arabic
    video_paths_by_action : dict
        Dictionary mapping action IDs to list of folder paths containing videos
    data_path : str
        Path where processed keypoints will be saved
    sequence_length : int
        Number of frames to extract from each video (default: 30)
    min_detection_confidence : float
        MediaPipe detection confidence threshold (default: 0.5)
    min_tracking_confidence : float
        MediaPipe tracking confidence threshold (default: 0.5)
    sanitize_folder_names : bool
        Whether to sanitize folder names (replace invalid chars with '-') (default: True)
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'total_actions_processed': Total number of actions processed
        - 'total_videos_processed': Total number of videos processed
        - 'total_frames_extracted': Total number of frames extracted
        - 'skipped_videos': Number of videos skipped due to errors
        - 'processing_summary': List of dicts with per-action statistics
    """
    import glob
    import re
    
    total_videos_processed = 0
    total_frames_extracted = 0
    skipped_videos = 0
    processing_summary = []
    
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=min_detection_confidence, 
                              min_tracking_confidence=min_tracking_confidence) as holistic:
        
        # Loop through each action
        for action_idx, (action, action_name) in enumerate(zip(actions, action_names)):
            print(f"\n[{action_idx+1}/{len(actions)}] Processing action: {action} - {action_name}")
            
            # Get all paths where this action exists (across all persons and splits)
            action_paths = video_paths_by_action.get(action, [])
            
            if not action_paths:
                print(f"  No folders found for action {action}")
                continue
            
            print(f"  Found {len(action_paths)} folders across persons/splits")
            
            # Sanitize action name for folder creation if needed
            folder_name = action_name
            if sanitize_folder_names:
                invalid_chars = r'[<>:"/\\|?*]'
                folder_name = re.sub(invalid_chars, '-', action_name).strip()
            
            # Global video counter for this action
            global_video_idx = 1
            action_videos_processed = 0
            action_frames_extracted = 0
            
            # Process videos from all locations
            for path_idx, action_folder in enumerate(action_paths, start=1):
                # Extract person and split info from path
                path_parts = action_folder.split(os.sep)
                person = next((p for p in path_parts if p in ['01', '02', '03']), 'unknown')
                split = next((s for s in path_parts if s in ['train', 'test']), 'unknown')
                
                print(f"  [{path_idx}/{len(action_paths)}] Person {person} - {split.upper()}: {action_folder}")
                
                # Get all video files in this action folder
                video_files = glob.glob(os.path.join(action_folder, '*.mp4')) + \
                              glob.glob(os.path.join(action_folder, '*.avi')) + \
                              glob.glob(os.path.join(action_folder, '*.mov'))
                
                if not video_files:
                    print(f"    No video files found")
                    continue
                    
                print(f"    Found {len(video_files)} videos")
                
                # Process each video
                for video_path in video_files:
                    video_name = os.path.basename(video_path).split('.')[0]
                    print(f"      Processing video {global_video_idx}: {video_name}")
                    
                    # Create folder for this sequence (use sanitized folder_name and global sequential number)
                    sequence_folder = os.path.join(data_path, folder_name, str(global_video_idx))
                    os.makedirs(sequence_folder, exist_ok=True)
                    
                    # Open video
                    cap = cv2.VideoCapture(video_path)
                    
                    # Get total frames in video
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    if total_frames == 0:
                        print(f"        Warning: Could not read frame count, skipping video")
                        cap.release()
                        skipped_videos += 1
                        continue
                    
                    # Calculate frame indices to sample (evenly distributed)
                    if total_frames >= sequence_length:
                        # Sample evenly across the video
                        frame_indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int)
                    else:
                        # If video is shorter than sequence_length, duplicate frames
                        frame_indices = np.arange(total_frames)
                        # Repeat frames to reach sequence_length
                        frame_indices = np.pad(frame_indices, (0, sequence_length - len(frame_indices)), 
                                              mode='edge')
                    
                    # Process only the selected frames
                    collected_frames = []
                    current_frame = 0
                    
                    for target_idx in frame_indices:
                        # Seek to target frame
                        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                        ret, frame = cap.read()
                        
                        if not ret:
                            print(f"        Warning: Could not read frame {target_idx}")
                            # Use zeros if frame read fails
                            keypoints = np.zeros(1662)
                        else:
                            # Make detections
                            image, results = mediapipe_detection(frame, holistic)
                            
                            # Extract keypoints
                            keypoints = extract_keypoints(results)
                        
                        # Save keypoints
                        npy_path = os.path.join(sequence_folder, f"{current_frame}.npy")
                        np.save(npy_path, keypoints)
                        collected_frames.append(keypoints)
                        current_frame += 1
                    
                    cap.release()
                    print(f"        Processed {len(collected_frames)}/{sequence_length} frames from {total_frames} total frames")
                    
                    action_videos_processed += 1
                    action_frames_extracted += len(collected_frames)
                    total_videos_processed += 1
                    total_frames_extracted += len(collected_frames)
                    global_video_idx += 1
            
            # Store summary for this action
            processing_summary.append({
                'action_id': action,
                'action_name': action_name,
                'videos_processed': action_videos_processed,
                'frames_extracted': action_frames_extracted
            })
    
    print("\n" + "=" * 70)
    print("DATASET PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"Total actions processed: {len(processing_summary)}")
    print(f"Total videos processed: {total_videos_processed}")
    print(f"Total frames extracted: {total_frames_extracted}")
    print(f"Skipped videos (errors): {skipped_videos}")
    print(f"Processed videos from all 3 persons (01, 02, 03) and both train/test splits")
    
    return {
        'total_actions_processed': len(processing_summary),
        'total_videos_processed': total_videos_processed,
        'total_frames_extracted': total_frames_extracted,
        'skipped_videos': skipped_videos,
        'processing_summary': processing_summary
    }


def process_karsl_videos_extract_keypoints_resume(actions, action_names, video_paths_by_action, 
                                                  data_path, sequence_length=30, 
                                                  min_detection_confidence=0.5, 
                                                  min_tracking_confidence=0.5,
                                                  sanitize_folder_names=True,
                                                  start_from_action=0,
                                                  end_at_action=None):
    """
    Process KArSL-502 dataset videos and extract keypoints using MediaPipe Holistic.
    
    This function:
    1. Processes videos from all persons (01, 02, 03) and splits (train, test)
    2. Extracts keypoints from each video using MediaPipe Holistic
    3. Saves keypoints as .npy files organized by action and sequence number
    4. Handles videos of varying lengths by sampling frames evenly
    5. Allows resuming from a specific action index
    
    Parameters:
    -----------
    actions : numpy.ndarray
        Array of action IDs (e.g., ['0001', '0002', ...])
    action_names : numpy.ndarray
        Array of action names in Arabic
    video_paths_by_action : dict
        Dictionary mapping action IDs to list of folder paths containing videos
    data_path : str
        Path where processed keypoints will be saved
    sequence_length : int
        Number of frames to extract from each video (default: 30)
    min_detection_confidence : float
        MediaPipe detection confidence threshold (default: 0.5)
    min_tracking_confidence : float
        MediaPipe tracking confidence threshold (default: 0.5)
    sanitize_folder_names : bool
        Whether to sanitize folder names (replace invalid chars with '-') (default: True)
    start_from_action : int
        Index of action to start processing from (default: 0)
    end_at_action : int or None
        Index of action to end processing at (exclusive). If None, process until the end (default: None)
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'total_actions_processed': Total number of actions processed
        - 'total_videos_processed': Total number of videos processed
        - 'total_frames_extracted': Total number of frames extracted
        - 'skipped_videos': Number of videos skipped due to errors
        - 'processing_summary': List of dicts with per-action statistics
        - 'start_from_action': Starting action index
        - 'end_at_action': Ending action index
    """
    import glob
    
    total_videos_processed = 0
    total_frames_extracted = 0
    skipped_videos = 0
    processing_summary = []
    
    # Validate start and end indices
    if start_from_action < 0 or start_from_action >= len(actions):
        print(f"Error: start_from_action ({start_from_action}) is out of range [0, {len(actions)-1}]")
        return None
    
    if end_at_action is None:
        end_at_action = len(actions)
    elif end_at_action <= start_from_action or end_at_action > len(actions):
        print(f"Error: end_at_action ({end_at_action}) must be > start_from_action ({start_from_action}) and <= {len(actions)}")
        return None
    
    print("\n" + "=" * 70)
    print(f"RESUMING PROCESSING FROM ACTION {start_from_action} TO {end_at_action - 1}")
    print("=" * 70)
    print(f"Processing range: {actions[start_from_action]} ({action_names[start_from_action]}) to {actions[end_at_action-1]} ({action_names[end_at_action-1]})")
    print(f"Total actions to process: {end_at_action - start_from_action}")
    print("=" * 70)
    
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=min_detection_confidence, 
                              min_tracking_confidence=min_tracking_confidence) as holistic:
        
        # Loop through each action in the specified range
        for action_idx in range(start_from_action, end_at_action):
            action = actions[action_idx]
            action_name = action_names[action_idx]
            
            print(f"\n[{action_idx+1}/{len(actions)}] (Processing {action_idx - start_from_action + 1}/{end_at_action - start_from_action}) Processing action: {action} - {action_name}")
            
            # Get all paths where this action exists (across all persons and splits)
            action_paths = video_paths_by_action.get(action, [])
            
            if not action_paths:
                print(f"  No folders found for action {action}")
                continue
            
            print(f"  Found {len(action_paths)} folders across persons/splits")
            
            # Sanitize action name for folder creation if needed
            folder_name = action_name
            if sanitize_folder_names:
                invalid_chars = r'[<>:"/\\|?*]'
                folder_name = re.sub(invalid_chars, '-', action_name).strip()
            
            # Global video counter for this action
            global_video_idx = 1
            action_videos_processed = 0
            action_frames_extracted = 0
            
            # Process videos from all locations
            for path_idx, action_folder in enumerate(action_paths, start=1):
                # Extract person and split info from path
                path_parts = action_folder.split(os.sep)
                person = next((p for p in path_parts if p in ['01', '02', '03']), 'unknown')
                split = next((s for s in path_parts if s in ['train', 'test']), 'unknown')
                
                print(f"  [{path_idx}/{len(action_paths)}] Person {person} - {split.upper()}: {action_folder}")
                
                # Get all video files in this action folder
                video_files = glob.glob(os.path.join(action_folder, '*.mp4')) + \
                              glob.glob(os.path.join(action_folder, '*.avi')) + \
                              glob.glob(os.path.join(action_folder, '*.mov'))
                
                if not video_files:
                    print(f"    No video files found")
                    continue
                    
                print(f"    Found {len(video_files)} videos")
                
                # Process each video
                for video_path in video_files:
                    video_name = os.path.basename(video_path).split('.')[0]
                    print(f"      Processing video {global_video_idx}: {video_name}")
                    
                    # Create folder for this sequence (use sanitized folder_name and global sequential number)
                    sequence_folder = os.path.join(data_path, folder_name, str(global_video_idx))
                    os.makedirs(sequence_folder, exist_ok=True)
                    
                    # Open video
                    cap = cv2.VideoCapture(video_path)
                    
                    # Get total frames in video
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    if total_frames == 0:
                        print(f"        Warning: Could not read frame count, skipping video")
                        cap.release()
                        skipped_videos += 1
                        continue
                    
                    # Calculate frame indices to sample (evenly distributed)
                    if total_frames >= sequence_length:
                        # Sample evenly across the video
                        frame_indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int)
                    else:
                        # If video is shorter than sequence_length, duplicate frames
                        frame_indices = np.arange(total_frames)
                        # Repeat frames to reach sequence_length
                        frame_indices = np.pad(frame_indices, (0, sequence_length - len(frame_indices)), 
                                              mode='edge')
                    
                    # Process only the selected frames
                    collected_frames = []
                    current_frame = 0
                    
                    for target_idx in frame_indices:
                        # Seek to target frame
                        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                        ret, frame = cap.read()
                        
                        if not ret:
                            print(f"        Warning: Could not read frame {target_idx}")
                            # Use zeros if frame read fails
                            keypoints = np.zeros(1662)
                        else:
                            # Make detections
                            image, results = mediapipe_detection(frame, holistic)
                            
                            # Extract keypoints
                            keypoints = extract_keypoints(results)
                        
                        # Save keypoints
                        npy_path = os.path.join(sequence_folder, f"{current_frame}.npy")
                        np.save(npy_path, keypoints)
                        collected_frames.append(keypoints)
                        current_frame += 1
                    
                    cap.release()
                    print(f"        Processed {len(collected_frames)}/{sequence_length} frames from {total_frames} total frames")
                    
                    action_videos_processed += 1
                    action_frames_extracted += len(collected_frames)
                    total_videos_processed += 1
                    total_frames_extracted += len(collected_frames)
                    global_video_idx += 1
            
            # Store summary for this action
            processing_summary.append({
                'action_id': action,
                'action_name': action_name,
                'videos_processed': action_videos_processed,
                'frames_extracted': action_frames_extracted
            })
    
    print("\n" + "=" * 70)
    print("DATASET PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"Total actions processed: {len(processing_summary)}")
    print(f"Total videos processed: {total_videos_processed}")
    print(f"Total frames extracted: {total_frames_extracted}")
    print(f"Skipped videos (errors): {skipped_videos}")
    print(f"Processed actions from index {start_from_action} to {end_at_action - 1}")
    
    return {
        'total_actions_processed': len(processing_summary),
        'total_videos_processed': total_videos_processed,
        'total_frames_extracted': total_frames_extracted,
        'skipped_videos': skipped_videos,
        'processing_summary': processing_summary,
        'start_from_action': start_from_action,
        'end_at_action': end_at_action
    }


def load_processed_keypoints(actions, action_names, data_path, sequence_length=30, 
                            start_action=0, end_action=None, include_null_class=False,
                            null_class_folder='null_no_sign'):
    """
    Load processed keypoint sequences from saved .npy files.
    
    This function loads the extracted keypoint data that was previously processed
    and saved by the process_karsl_videos_extract_keypoints() function. It reads
    all sequence folders for each action and loads the keypoint arrays.
    
    Parameters:
    -----------
    actions : numpy.ndarray
        Array of action IDs (e.g., ['0001', '0002', ...])
    action_names : numpy.ndarray
        Array of action names in Arabic
    data_path : str
        Path where processed keypoints are stored (e.g., 'MP_Data_KArSL')
    sequence_length : int
        Expected number of frames per sequence (default: 30)
    start_action : int
        Index of first action to load (default: 0)
    end_action : int or None
        Index of last action to load (exclusive). If None, load until the end (default: None)
    include_null_class : bool
        Whether to include null class (no sign) data (default: False)
    null_class_folder : str
        Folder name containing null class data (default: 'null_no_sign')
    
    Returns:
    --------
    tuple : (sequences, labels)
        - sequences: list of keypoint sequences, each shape (sequence_length, 1662)
        - labels: list of integer labels corresponding to action indices (0 to num_loaded_actions-1)
    
    Example:
    --------
    >>> # Load all actions
    >>> sequences, labels = load_processed_keypoints(actions, action_names, DATA_PATH, 30)
    >>> print(f"Loaded {len(sequences)} sequences")
    >>> 
    >>> # Load first 100 actions (0-99)
    >>> sequences, labels = load_processed_keypoints(actions, action_names, DATA_PATH, 30, 
    ...                                               start_action=0, end_action=100)
    >>> 
    >>> # Load actions 50-150
    >>> sequences, labels = load_processed_keypoints(actions, action_names, DATA_PATH, 30,
    ...                                               start_action=50, end_action=150)
    """
    # Validate start and end indices
    if start_action < 0 or start_action >= len(actions):
        print(f"Error: start_action ({start_action}) is out of range [0, {len(actions)-1}]")
        return [], []
    
    if end_action is None:
        end_action = len(actions)
    elif end_action <= start_action or end_action > len(actions):
        print(f"Error: end_action ({end_action}) must be > start_action ({start_action}) and <= {len(actions)}")
        return [], []
    
    print("\n" + "=" * 70)
    print(f"LOADING ACTIONS FROM INDEX {start_action} TO {end_action - 1}")
    print("=" * 70)
    print(f"Loading range: {actions[start_action]} ({action_names[start_action]}) to {actions[end_action-1]} ({action_names[end_action-1]})")
    print(f"Total actions to load: {end_action - start_action}")
    print("=" * 70 + "\n")
    
    sequences, labels = [], []
    
    # Use relative label indexing (0 to num_loaded_actions-1)
    for relative_idx, action_idx in enumerate(range(start_action, end_action)):
        action = actions[action_idx]
        action_name = action_names[action_idx]
        
        print(f"Loading action {action_idx+1}/{len(actions)} (subset: {relative_idx + 1}/{end_action - start_action}): {action} - {action_name}")
        
        # Sanitize folder name to match how folders were created (replace invalid chars with '-')
        invalid_chars = r'[<>:"/\\|?*]'
        sanitized_name = re.sub(invalid_chars, '-', action_name).strip()
        action_path = os.path.join(data_path, sanitized_name)
        
        # Check if folder exists
        if not os.path.exists(action_path):
            print(f"  ⚠ Skipping: Folder not found - '{action_name}'")
            continue
        
        # Get all sequence folders for this action (numbered 1, 2, 3, ...)
        try:
            sequence_folders = [f for f in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, f))]
        except (FileNotFoundError, OSError) as e:
            print(f"  ⚠ Skipping: Error accessing folder - {e}")
            continue
        
        # Sort numerically instead of alphabetically
        sequence_folders.sort(key=lambda x: int(x) if x.isdigit() else float('inf'))
        
        for sequence_folder in sequence_folders:
            sequence_path = os.path.join(action_path, sequence_folder)
            
            # Get all .npy files in this sequence
            frame_files = sorted([f for f in os.listdir(sequence_path) if f.endswith('.npy')],
                               key=lambda x: int(x.split('.')[0]))
            
            if len(frame_files) == 0:
                continue
                
            # Load all frames for this sequence
            window = []
            for frame_file in frame_files[:sequence_length]:  # Use first sequence_length frames
                frame_path = os.path.join(sequence_path, frame_file)
                frame_data = np.load(frame_path)
                window.append(frame_data)
            
            # Pad if necessary (if video has fewer frames than sequence_length)
            while len(window) < sequence_length:
                window.append(np.zeros(1662))  # Pad with zeros
            
            sequences.append(window)
            labels.append(relative_idx)  # Use relative index (0 to num_loaded_actions-1)
    
    # Load null class if enabled
    if include_null_class:
        print(f"\n{'='*70}")
        print(f"Loading NULL CLASS (no sign) data")
        print(f"{'='*70}")
        
        null_class_path = os.path.join(data_path, null_class_folder)
        
        if os.path.exists(null_class_path):
            null_class_label = end_action - start_action  # Assign next available label
            
            try:
                sequence_folders = [f for f in os.listdir(null_class_path) 
                                  if os.path.isdir(os.path.join(null_class_path, f))]
                sequence_folders.sort(key=lambda x: int(x) if x.isdigit() else float('inf'))
                
                null_sequences_count = 0
                
                for sequence_folder in sequence_folders:
                    sequence_path = os.path.join(null_class_path, sequence_folder)
                    
                    frame_files = sorted([f for f in os.listdir(sequence_path) if f.endswith('.npy')],
                                       key=lambda x: int(x.split('.')[0]))
                    
                    if len(frame_files) == 0:
                        continue
                    
                    window = []
                    for frame_file in frame_files[:sequence_length]:
                        frame_path = os.path.join(sequence_path, frame_file)
                        frame_data = np.load(frame_path)
                        window.append(frame_data)
                    
                    while len(window) < sequence_length:
                        window.append(np.zeros(1662))
                    
                    sequences.append(window)
                    labels.append(null_class_label)
                    null_sequences_count += 1
                
                print(f"✓ Loaded {null_sequences_count} null class sequences")
                print(f"  Null class label: {null_class_label}")
                
            except Exception as e:
                print(f"⚠ Error loading null class data: {e}")
        else:
            print(f"⚠ Warning: Null class folder not found: {null_class_path}")
            print(f"  Run record_null_class.py to create null class data")
    
    print(f"\n{'='*70}")
    print(f"Total sequences loaded: {len(sequences)}")
    print(f"Total labels: {len(labels)}")
    if sequences:
        print(f"Sequences shape: {np.array(sequences).shape}")
        print(f"Labels shape: {np.array(labels).shape}")
        print(f"Label range: {min(labels) if labels else 'N/A'} to {max(labels) if labels else 'N/A'}")
        if include_null_class:
            print(f"Classes: {end_action - start_action} KArSL actions + 1 null class = {end_action - start_action + 1} total")
    print(f"{'='*70}")
    
    return sequences, labels




