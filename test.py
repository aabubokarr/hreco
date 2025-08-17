import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import mediapipe as mp
import argparse

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, 
                      max_num_hands=1, 
                      min_detection_confidence=0.5)

# Load the trained model and labels
model = load_model("asl.h5")

class_labels = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
                'L', 'M', 'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 
                'W', 'X', 'Y', 'Z']

# Initialize LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(class_labels)

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

def predict_asl(image_path):
    """Predict ASL sign from an image file"""
    landmarks = extract_hand_landmarks(image_path)
    if landmarks is None:
        print("No hand detected in the image.")
        return
    
    landmarks = landmarks.reshape(1, -1)
    prediction = model.predict(landmarks, verbose=0)
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    confidence = np.max(prediction)
    
    print(f"Predicted ASL Sign: {predicted_class}")
    print(f"Confidence {confidence}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ASL Sign Recognition from Images')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    args = parser.parse_args()
    
    predict_asl(args.image_path)
    
    # Release MediaPipe resources
    hands.close()