# model_utils.py

import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

# Initialize MediaPipe Holistic model and drawing utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Load the pre-trained Keras model
model = load_model('action.h5')  # Ensure the model path is correct

# Define the actions you want to recognize
actions = ['hello', 'imagine', 'cup']  # Update as needed

# Function to process image with MediaPipe
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Function to draw styled landmarks
def draw_styled_landmarks(image, results):
    # Draw face landmarks
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(159, 176, 243), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(173, 216, 230), thickness=1, circle_radius=1)
                             )
    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(159, 176, 243), thickness=2, circle_radius=2), 
                             mp_drawing.DrawingSpec(color=(173, 216, 230), thickness=2, circle_radius=1)
                             ) 
    # Draw hand landmarks
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(159, 176, 243), thickness=2, circle_radius=2), 
                             mp_drawing.DrawingSpec(color=(173, 216, 230), thickness=2, circle_radius=1)
                             ) 
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(159, 176, 243), thickness=2, circle_radius=2), 
                             mp_drawing.DrawingSpec(color=(173, 216, 230), thickness=2, circle_radius=1)
                             ) 

# Function to extract keypoints
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Function to predict the action
def predict_action(sequence):
    if len(sequence) == 30:  # Ensure we have a sequence of 30 frames
        sequence_array = np.expand_dims(np.array(sequence), axis=0)
        predictions = model.predict(sequence_array)[0]
        action = actions[np.argmax(predictions)]
        confidence = np.max(predictions)
        return action, confidence
    return None, None
