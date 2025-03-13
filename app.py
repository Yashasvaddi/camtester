from flask import Flask, render_template, Response
import cv2
import time as t
import math
import mediapipe as mp
import waitress

app = Flask(__name__)

# Open the webcam
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Set the desired frame size
frame_width = 640
frame_height = 480
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# Resize the frame
def resize_frame(frame):
    return cv2.resize(frame, (frame_width, frame_height))

# Convert the frame to RGB (for better web rendering)
def convert_to_rgb(frame):
    return frame

# Function to generate video frames for streaming
def generate_frames():
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize the frame and convert it to RGB
            frame_resized = resize_frame(frame)
            frame_rgb = convert_to_rgb(frame_resized)
            # frame_rgb=cv2.flip(frame_rgb,1)
            # Encode the frame to JPEG
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            timestamp=t.time()
            formatted_time1=t.strftime("%Y-%m-%d",t.localtime(timestamp))
            formatted_time2=t.strftime("%H:%M:%S",t.localtime(timestamp))
            cv2.putText(frame_rgb,formatted_time2,(10,25),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
            cv2.putText(frame_rgb,formatted_time1,(10,45),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0,0, 255), 2)

            _, buffer = cv2.imencode('.jpg', frame_rgb)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

# Home route to display the webpage
@app.route('/')
def index():
    return render_template('index.html')

# Video stream route to stream frames
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
