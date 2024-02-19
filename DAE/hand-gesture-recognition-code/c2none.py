import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
from tensorflow.keras.models import load_model

# Function for Text-to-Speech
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
with open('gesture.names', 'r') as f:
    classNames = f.read().split('\n')

cap = cv2.VideoCapture(0)

# State machine variables
current_gesture = None
previous_gesture = None

# Frame counter for skipping frames
frame_counter = 0

# Flag to indicate whether TTS is currently speaking
tts_speaking = False

# Coordinates of the "none box"
none_box_x_start, none_box_y_start = 350, 100
none_box_x_end, none_box_y_end = 100, 300

while True:
    _, frame = cap.read()

    frame_counter += 1
    if frame_counter % 5 != 0:  # Skip model inference every 5 frames
        continue

    x, y, _ = frame.shape
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)

    className = ''

    # Draw the "none box"
    cv2.rectangle(frame, (none_box_x_start, none_box_y_start), (none_box_x_end, none_box_y_end), (0, 255, 0), 2)

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                # Check if the hand landmark is within the "none box" region
                if none_box_x_start < lmx < none_box_x_end and none_box_y_start < lmy < none_box_y_end:
                    continue  # Skip landmarks within the "none box"

                landmarks.append([lmx, lmy])

            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            prediction = model.predict([landmarks])
            confidence = np.max(prediction)
            classID = np.argmax(prediction)

            if confidence > 0.5:  # Set a confidence threshold
                className = classNames[classID]

                # Update gestures and handle transitions
                previous_gesture = current_gesture
                current_gesture = className

                if current_gesture != previous_gesture:
                    # Use threading for TTS to avoid blocking the main thread
                    if not tts_speaking:
                        tts_speaking = True
                        text_to_speech(current_gesture)
                        tts_speaking = False

    cv2.putText(frame, current_gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Output", frame)

    # Reset the frame counter after skipping frames
    if frame_counter > 1000:
        frame_counter = 0

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
