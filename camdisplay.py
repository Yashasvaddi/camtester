import streamlit as st
import cv2

cap=cv2.VideoCapture(0)
placeholder=st.empty()
while(True):
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    placeholder.image(frame,channels="RGB")
    # cv2.imshow("frame",frame)
    if(cv2.waitKey(1)==ord('x')):
        break