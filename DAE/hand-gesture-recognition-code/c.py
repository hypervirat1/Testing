import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import concurrent.futures
from tensorflow.keras.models import load_model

# Function for Text-to-Speech
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Initialize mediapipe
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

# Flag to indicate whether TTS is currently speaking
tts_speaking = False

def process_frames():
    global cap, previousClassName, tts_speaking

    frame_counter = 0

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
                    # Use threading for TTS to avoid blocking the main thread
                    if not tts_speaking:
                        tts_speaking = True
                        text_to_speech(className)
                        previousClassName = className
                        tts_speaking = False

        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Output", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Use a thread pool for frame processing
with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.submit(process_frames)
