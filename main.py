import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the trained model and labels
model = load_model("asl.h5")

class_labels = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
                'L', 'M', 'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 
                'W', 'X', 'Y', 'Z']

# Initialize LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(class_labels)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Initialize webcam
cap = cv2.VideoCapture()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    
    # Flip the image horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect hands
    results = hands.process(frame_rgb)
    
    # Draw hand landmarks and make predictions
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            landmarks = np.array(landmarks).reshape(1, -1)
            
            try:
                # Predict ASL letter
                prediction = model.predict(landmarks, verbose=0)
                predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]
                confidence = np.max(prediction)
                
                # Display prediction
                cv2.putText(frame, f"{predicted_class} ({confidence:.2f})", 
                           (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                           3, (0, 255, 0), 5)
                
            except Exception as e:
                print(f"Prediction error: {e}")
            
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76)),  # Hand landmarks color
                mp_drawing.DrawingSpec(color=(250, 44, 250)))  # Connections color
    
    # Display the resulting frame
    cv2.imshow('Hreco', frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
