import streamlit as st
import cv2
import numpy as np

st.title("Live Webcam Feed")

# Create a placeholder for the image
placeholder = st.empty()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        st.warning("Failed to capture frame. Check your webcam connection.")
        break

    frame = cv2.flip(frame, 1)  # Flip for a better user experience
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

    # Update the placeholder with the new frame
    placeholder.image(frame, channels="RGB")

cap.release()
