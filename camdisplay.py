from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# Open the webcam
cap = cv2.VideoCapture(3)

# Set the desired frame size
frame_width = 640
frame_height = 480

# Resize the frame
def resize_frame(frame):
    return cv2.resize(frame, (frame_width, frame_height))

# Convert the frame to RGB (for better web rendering)
def convert_to_rgb(frame):
    return frame

# Function to generate video frames for streaming
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize the frame and convert it to RGB
        frame_resized = resize_frame(frame)
        frame_rgb = convert_to_rgb(frame_resized)
        # frame_rgb=cv2.flip(frame_rgb,1)
        # Encode the frame to JPEG
        cv2.putText(frame_rgb,"hello how are you?",(10,300),cv2.FONT_HERSHEY_COMPLEX,1.8,(255,255,255),1,cv2.LINE_AA)
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
