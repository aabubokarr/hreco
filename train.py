import os
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, 
                      max_num_hands=1, 
                      min_detection_confidence=0.5)

def extract_hand_landmarks(image_path):
    """Extract hand landmarks from an image using MediaPipe"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        landmarks = []
        for landmark in results.multi_hand_landmarks[0].landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])  # 3D coordinates
        return np.array(landmarks)  # Shape: (63,)
    return None

# Main processing
X = []
y = []

dataset_path = "asl_dataset"
valid_classes = [d for d in os.listdir(dataset_path) 
               if os.path.isdir(os.path.join(dataset_path, d)) 
               and not d.startswith('.')]

for class_name in tqdm(valid_classes, desc="Processing classes"):
    class_dir = os.path.join(dataset_path, class_name)
    for image_name in os.listdir(class_dir):
        if image_name.startswith('.'):  # Skip hidden files
            continue
        image_path = os.path.join(class_dir, image_name)
        landmarks = extract_hand_landmarks(image_path)
        if landmarks is not None:
            X.append(landmarks)
            y.append(class_name)

# Save the data
np.save("X.npy", np.array(X))
np.save("y.npy", np.array(y))

# Release MediaPipe resources
hands.close()
