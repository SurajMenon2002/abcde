# app.py

from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
from model_utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints, predict_action

app = Flask(__name__)

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic

# Define the video stream generator function
def gen_frames():
    cap = cv2.VideoCapture(0)  # Use the default webcam
    
    target_width = 300
    target_height = 300
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
    
    min_frames_for_prediction = 15
    max_sequence_length = 30
    sequence = []

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            frame = cv2.flip(frame, 1)

            # Process the frame with MediaPipe
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            # Extract keypoints and update the sequence
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-max_sequence_length:]
            
            # Check if we have enough frames for an early prediction
            if len(sequence) >= min_frames_for_prediction:
                action, confidence = predict_action(sequence)
                if action is not None:
                    cv2.putText(image, f'{action} ({confidence:.2f})', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Route for rendering the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Route for video streaming
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/exit')
def exit_page():
    return render_template('exit.html')
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
