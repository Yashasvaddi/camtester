import streamlit as st
import cv2
import time
from streamlit_webrtc import webrtc_streamer
webrtc_streamer(key="camera")
st.title("Live Webcam Feed")

# Function to check camera access
def check_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("⚠️ Camera access denied! Please grant permission and restart.")
        return False
    cap.release()
    return True

# Create a button to start the camera
if st.button("Start Camera"):
    if check_camera():
        cap = cv2.VideoCapture(0)
        placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to capture frame. Try restarting the app.")
                break

            frame = cv2.flip(frame, 1)  # Flip for a better user experience
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

            # Display the frame using Streamlit
            placeholder.image(frame, channels="RGB")

            # Add a Stop button to exit
            if st.button("Stop Camera"):
                break

        cap.release()
