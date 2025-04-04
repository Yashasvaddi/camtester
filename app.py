from flask import Flask, render_template, Response
import cv2
import time
import mediapipe as mp
import google.generativeai as genai
import pyttsx3
import speech_recognition as sr
import os
import math as m
import PIL.Image
import threading as th


# Initialize Flask app
app = Flask(__name__)

# Configure Gemini AI
genai.configure(api_key='AIzaSyC3vNkSnEJl-eFloSm9M4Bw0F_cJv2vusY')
model = genai.GenerativeModel("gemini-2.0-flash")

# Initialize Camera
cap = cv2.VideoCapture(0)

# Initialize Mediapipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Create directory for saving faces
face_save_path = "C:\\New folder\\codes\\college stuff\\camtester\\faces"
if not os.path.exists(face_save_path):
    os.makedirs(face_save_path)

# Global Variables
frame_width, frame_height = 640, 480
answer = ""
caption = ""
thread_flag = True
face_count = 0  # Counter for naming saved face images
flag=False
option_changed = th.Event()
Option=0
flag1=True
def selector():
    global Option
    if Option<4:
        Option=Option+1
        print(Option)
    else:
        Option=1
    option_changed.set()

def wrap_text(text, max_width, font, font_scale, thickness):
    """Splits text into multiple lines to fit inside OpenCV window."""
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = current_line + " " + word if current_line else word
        text_size, _ = cv2.getTextSize(test_line, font, font_scale, thickness)

        if text_size[0] > max_width:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test_line

    lines.append(current_line)  # Add the last line
    return lines

def image_prcoess():
    global answer
    print("Image being sent to gemini")
    image_path='C:\\New folder\\codes\\college stuff\\camtester\\frames\\frame.jpg'
    image=PIL.Image.open(image_path)
    response=model.generate_content([image,"What is this? Answer in one line. If there is some text on the object idetify the object and read the text"])
    answer=response.text
# Menu Function
def menu():
    option=0
    global thread_flag, answer, caption, flag, Option,flag1
    recognizer = sr.Recognizer()
    engine = pyttsx3.init()
    engine.say("Hello")
    engine.runAndWait()
    print("Menu thread started")
    last_option=0
    while thread_flag:
        try:
            with sr.Microphone() as source:
                if option_changed.is_set() or Option != last_option:
                            option = Option
                            last_option = option
                            print(f"Option changed to {option}")
                            
                            # Handle the option change
                            if option == 1:
                                engine.say("Assistant at your service")
                                engine.runAndWait()
                                option=0
                            elif option == 2:
                                engine.say("Live captioning started")
                                engine.runAndWait()
                                option=0
                            elif option == 3:
                                engine.say("Identification Mode activated")
                                engine.runAndWait()
                                option=0
                                flag = True
                            elif option == 4:
                                engine.say("Search Mode activated")
                                engine.runAndWait()
                                option=0
                                flag1 = True
                                
                            option_changed.clear()  # Reset the event
                print("Listening....")
                audio = recognizer.listen(source)
                question = recognizer.recognize_google(audio)

                if question.strip():
                    print(question)
                
                if question.lower() == "exit":
                    engine.say("Exiting")
                    engine.runAndWait()
                    break
                
                if question.lower() == 'assistant' or option==1:
                    engine.say("Assistant at your service")
                    engine.runAndWait()
                    print("Assistant thread started")
                    while thread_flag:
                        try:
                            with sr.Microphone() as source:
                                print("Assistant Listening....")
                                audio = recognizer.listen(source)
                                question = recognizer.recognize_google(audio) + " Answer in one line"

                                if question.strip():
                                    print(question)
                                
                                if "exit" in question.lower():
                                    engine.say("Exiting")
                                    engine.runAndWait()
                                    break
                                
                                response = model.generate_content(question)
                                answer = response.text
                                
                                engine.say(answer)
                                engine.runAndWait()
                        except sr.UnknownValueError:
                            print("Unable to understand")
                        except OSError as e:
                            print(f"Audio error: {e}")
                            break
                    print("Assistant thread closed")
                    answer = ''
                
                if question.lower() == 'caption' or option==2:
                    engine.say("Live captioning started")
                    engine.runAndWait()
                    print("Captioning thread started")
                    while thread_flag:
                        try:
                            with sr.Microphone() as source:
                                print("Live Caption Listening....")
                                audio = recognizer.listen(source)
                                question = recognizer.recognize_google(audio)
                                if question.strip():
                                    print(question)
                                if "exit" in question.lower():
                                    engine.say("Exiting")
                                    engine.runAndWait()
                                    break
                                caption = question
                        except sr.UnknownValueError:
                            print("Unable to understand")
                        except OSError as e:
                            print(f"Audio error: {e}")
                            break
                    print("Assistant thread closed")
                    caption = ''
                if question.lower()=='identify' or option==3:
                    engine.say("Identification Mode activated")
                    engine.runAndWait()
                    flag=True
                if question.lower()=='stop' and flag:
                    engine.say("Deactivating identification mode")
                    engine.runAndWait()
                    flag=False
                if question.lower()=='find' or option==4:
                    engine.say("Search Mode activated")
                    engine.runAndWait()
                    flag1=True
                if question.lower()=='stop' and flag1:
                    engine.say("Deactivating identification mode")
                    engine.runAndWait()
                    flag1=False
        except sr.UnknownValueError:
            print("Unable to understand")
        except OSError as e:
            print(f"Audio error: {e}")
            break
    print("Menu thread closed")

# Resize the frame
def resize_frame(frame):
    return cv2.resize(frame, (frame_width, frame_height))

# Video Stream Function
# Video Stream Function
def generate_frames():
    global answer, thread_flag, caption, face_count, flag, captured_face, captured_frame,flag1
    done_flag = False
    reflagger = False
    reset_time = 0
    captured_face = None
    captured_frame = None  # Initialize captured_frame
    temp_flag=True
    selector_flag=True
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
         mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break  # Exit if camera read fails

            # Resize and convert frame
            frame_rgb = resize_frame(frame)
            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)

            # Process for hand detection
            results_hands = hands.process(frame_rgb)
            
            # Process for face detection
            results_faces = face_detection.process(frame_rgb)
            
            # Convert back to BGR for OpenCV display
            frame_display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Initialize face bounding box coordinates
            x, y, w_box, h_box = 0, 0, 0, 0
            face_detected = False
            hand_detected=False
            # Draw hand landmarks if detected
            if results_hands.multi_hand_landmarks:
                hand_detected=True
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame_display, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    # If face capture is enabled
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    thumb_x = int(thumb_tip.x * frame_display.shape[1])
                    thumb_y = int(thumb_tip.y * frame_display.shape[0])
                    
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    index_x = int(index_tip.x * frame_display.shape[1])
                    index_y = int(index_tip.y * frame_display.shape[0])
                    cv2.rectangle(frame_display,(80,280),(160,340),(0,0,0),1)
                    distance = m.sqrt((index_x - thumb_x)**2 + (index_y - thumb_y)**2)
                    if 80<=index_x<=160 and 280<=index_y<=340 and selector_flag:
                        selector()
                        print("Option switched")
                        selector_flag=False
                    if not (80<=index_x<=160 and 280<=index_y<=340) and not selector_flag:
                        selector_flag=True
                        
                    if flag and results_faces.detections:
                        # Get thumb and index finger position
                        
                        # Draw line between thumb and index finger
                        cv2.line(frame_display, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 2)
                        
                        # Calculate distance between thumb and index finger
                        distance = m.sqrt((index_x - thumb_x)**2 + (index_y - thumb_y)**2)
                        
                        # Display distance
                        cv2.putText(frame_display, f"Distance: {distance:.1f}", 
                                   (10, 60), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
                        
                        # Check if fingers are pinched (distance < 30) and not already captured
                        if distance < 30 and not reflagger:
                            caption = "Face captured!"
                            reflagger = True  # Prevent multiple captures
                            done_flag = True
                            
                            # Set a timer to reset the reflagger after 2 seconds
                            reset_time = time.time() + 2
            
            # Process face detection
            if results_faces.detections:
                face_detected = True
                for detection in results_faces.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame_display.shape
                    
                    # Calculate absolute coordinates
                    x = max(0, int(bboxC.xmin * iw))
                    y = max(0, int(bboxC.ymin * ih))
                    w_box = min(int(bboxC.width * iw), iw - x)
                    h_box = min(int(bboxC.height * ih), ih - y)
                    
                    # Draw rectangle around face
                    cv2.rectangle(frame_display, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                    
                    # If we need to capture the face
                    if done_flag and face_detected:
                        # Crop the face
                        face = frame[y:y+h_box, x:x+w_box]
                        
                        if face.size != 0:
                            # Save the face
                            face_filename = os.path.join(face_save_path, "face.jpg")
                            cv2.imwrite(face_filename, face)
                            
                            # Load the saved face for display
                            captured_face = cv2.imread(face_filename)
                            done_flag = False
            
            # If no face detected, capture the entire frame
            if not face_detected and hand_detected and flag1:
                captured_frame = frame.copy()
                frame_filename=os.path.join('C:\\New folder\\codes\\college stuff\\camtester\\frames',"frame.jpg")
                cv2.imwrite(frame_filename,captured_frame)
                if temp_flag and flag1:
                    t1=th.Thread(target=image_prcoess,daemon=True)
                    t1.start()
                    temp_flag=False
                    t1.join()

                if t1 is not None and not t1.is_alive():
                    temp_flag = True


            # Reset reflagger after timeout
            if reflagger and time.time() > reset_time:
                reflagger = False
                caption = "Pinch fingers to capture face"
            
            # Overlay the captured face if available
            if captured_face is not None:
                try:
                    # Resize captured face for overlay
                    face_height, face_width = 150, 150
                    resized_face = cv2.resize(captured_face, (face_width, face_height))
                    
                    # Position for overlay (top-right corner with padding)
                    x_offset = frame_display.shape[1] - face_width - 20
                    y_offset = 20
                    
                    # Create a white border around the overlay
                    cv2.rectangle(frame_display, 
                                 (x_offset-3, y_offset-3), 
                                 (x_offset+face_width+3, y_offset+face_height+3), 
                                 (255, 255, 255), 2)
                    
                    # Simple overlay method
                    frame_display[y_offset:y_offset+face_height, x_offset:x_offset+face_width] = resized_face


                    if not face_detected and captured_frame is not None:
                    # Resize the captured frame for display in a corner
                        captured_frame_resized = cv2.resize(captured_frame, (int(frame_display.shape[1] * 0.3), int(frame_display.shape[0] * 0.3)))
                        
                        # Position for overlay (top-left corner)
                        x_offset = 20
                        y_offset = 20

                        # Create a white border around the overlay
                        cv2.rectangle(frame_display, 
                                    (x_offset-3, y_offset-3), 
                                    (x_offset+captured_frame_resized.shape[1]+3, y_offset+captured_frame_resized.shape[0]+3), 
                                    (255, 255, 255), 2)
                        
                        # Overlay the captured frame
                        frame_display[y_offset:y_offset+captured_frame_resized.shape[0], x_offset:x_offset+captured_frame_resized.shape[1]] = captured_frame_resized
                except Exception as e:
                    print(f"Error overlaying face: {e}")

            # If no face detected, display the captured frame in the corne

            # Add text overlays
            cv2.rectangle(frame_display,(10,400),(630,480),(255,255,255),-1)
            wrapped_lines = wrap_text(answer, max_width=720, font=cv2.FONT_HERSHEY_COMPLEX, font_scale=1.0, thickness=1)
            y = 416 
            for line in wrapped_lines:
                cv2.putText(frame_display, line, (20, y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                y += 25
            cv2.putText(frame_display, caption, (50, 416), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            # Add timestamp
            timestamp = time.time()
            formatted_time1 = time.strftime("%Y-%m-%d", time.localtime(timestamp))
            formatted_time2 = time.strftime("%H:%M:%S", time.localtime(timestamp))
            cv2.rectangle(frame_display,(505,10),(675,55),(0,0,0),-1)
            cv2.putText(frame_display, formatted_time2, (520, 25), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame_display, formatted_time1, (520, 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            cv2.rectangle(frame_display,(80,280),(160,340),(0,0,0),1)
            # Encode frame for streaming
            _, buffer = cv2.imencode('.jpg', frame_display)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    thread_flag = True
    th.Thread(target=menu, daemon=True).start()
    app.run(debug=True, threaded=True, use_reloader=True)