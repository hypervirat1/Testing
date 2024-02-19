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

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
with open('gesture.names', 'r') as f:
    classNames = f.read().split('\n')
print(classNames)

cap = cv2.VideoCapture(0)

previousClassName = ''  # Variable to store the previous class name

while True:
    _, frame = cap.read()

    x, y, c = frame.shape
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)

    className = ''

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            prediction = model.predict([landmarks])
            classID = np.argmax(prediction)
            className = classNames[classID]

            # Check if the class name has changed
            if className != previousClassName:
                previousClassName = className

    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Output", frame)

    try:
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord(' '):  # Check for space bar press
            print("Space bar pressed")
            if className:
                print("Recognized gesture:", className)
                text_to_speech(className)  # Read the text
    except Exception as e:
        print("Exception:", e)

cap.release()
cv2.destroyAllWindows()
