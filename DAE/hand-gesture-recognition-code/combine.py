import tkinter as tk
from PIL import Image, ImageTk



def voice():
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

    # Add a frame counter
    frame_counter = 0

    # Flag to indicate whether TTS is currently speaking
    tts_speaking = False

    def process_frames():
        global cap, frame_counter, previousClassName, tts_speaking

        while True:
            _, frame = cap.read()

            frame_counter += 1
            if frame_counter % 5 != 0:  # Skip model inference every 5 frames
                continue

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
                        # Use threading for TTS to avoid blocking the main thread
                        if not tts_speaking:
                            tts_speaking = True
                            text_to_speech(className)
                            previousClassName = className
                            tts_speaking = False

            cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("Output", frame)

            # Reset the frame counter after skipping frames
            if frame_counter > 1000:
                frame_counter = 0

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Use a thread pool for frame processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(process_frames)

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
                    # Use threading for TTS to avoid blocking the main thread
                    if not tts_speaking:
                        tts_speaking = True
                        text_to_speech(className)
                        previousClassName = className
                        tts_speaking = False

        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Output", frame)

        try:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print("Exception:", e)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


def paint():
    import mediapipe as mp
    import cv2
    import numpy as np
    import time
    from collections import deque
    import tensorflow.keras.models
    import pyttsx3
    # Constants

    ml = 150
    max_x, max_y = 250 + ml, 50
    curr_tool = "select tool"
    time_init = True
    rad = 40
    var_inits = False
    thick = 4
    prevx, prevy = 0, 0

    # Get tools function
    def getTool(x):
        if x < 50 + ml:
            return "line"
        elif x < 100 + ml:
            return "HandSign"
        elif x < 150 + ml:
            return "draw"
        elif x < 200 + ml:
            return "circle"
        else:
            return "erase"

    def index_raised(yi, y9):
        if (y9 - yi) > 40:
            return True
        return False

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # try1
    # Here is code for Canvas setup
    paintWindow = np.zeros((471, 636, 3)) + 255
    paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
    paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), (255, 0, 0), 2)
    paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), (0, 255, 0), 2)
    paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), (0, 0, 255), 2)
    paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), (0, 255, 255), 2)

    cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

    # Initialize the media pipe hands module
    hands = mp.solutions.hands
    hand_landmark = hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)
    draw = mp.solutions.drawing_utils

    # Drawing tools
    tools = cv2.imread("C:/Users/Shruti/Downloads/tools.png")
    tools = tools.astype('uint8')

    mask = np.ones((480, 640)) * 255
    mask = mask.astype('uint8')

    while True:
        _, frm = cap.read()
        frm = cv2.flip(frm, 1)
        rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

        op = hand_landmark.process(rgb)

        if op.multi_hand_landmarks:
            for i in op.multi_hand_landmarks:
                draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS)
                x, y = int(i.landmark[8].x * 640), int(i.landmark[8].y * 480)
                x, y = int(i.landmark[7].x * 640), int(i.landmark[7].y * 480)

                if x < max_x and y < max_y and x > ml:
                    if time_init:
                        ctime = time.time()
                        time_init = False
                    ptime = time.time()

                    cv2.circle(frm, (x, y), rad, (0, 255, 255), 2)
                    rad -= 1

                    if (ptime - ctime) > 0.8:
                        curr_tool = getTool(x)
                        print("Your current tool is set to: ", curr_tool)
                        time_init = True
                        rad = 40
                else:
                    time_init = True
                    rad = 40

                if curr_tool == "draw":
                    bpoints = [deque(maxlen=1024)]
                    gpoints = [deque(maxlen=1024)]
                    rpoints = [deque(maxlen=1024)]
                    ypoints = [deque(maxlen=1024)]

                    # These indexes will be used to mark the points in particular arrays of specific colour
                    blue_index = 0
                    green_index = 0
                    red_index = 0
                    yellow_index = 0

                    # The kernel to be used for dilation purpose
                    kernel = np.ones((5, 5), np.uint8)

                    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
                    colorIndex = 0

                    # initialize mediapipe
                    mpHands = mp.solutions.hands
                    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
                    mpDraw = mp.solutions.drawing_utils

                    # Initialize the webcam
                    cap = cv2.VideoCapture(0)
                    ret = True
                    while ret:
                        # Read each frame from the webcam
                        ret, frame = cap.read()

                        x, y, c = frame.shape

                        # Flip the frame vertically
                        frame = cv2.flip(frame, 1)
                        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
                        frame = cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), 2)
                        frame = cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 0), 2)
                        frame = cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 255), 2)
                        frame = cv2.rectangle(frame, (505, 1), (600, 65), (0, 255, 255), 2)
                        cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2,
                                    cv2.LINE_AA)
                        # frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                        # Get hand landmark prediction
                        result = hands.process(framergb)

                        # post process the result
                        if result.multi_hand_landmarks:
                            landmarks = []
                            for handslms in result.multi_hand_landmarks:
                                for lm in handslms.landmark:
                                    # # print(id, lm)
                                    # print(lm.x)
                                    # print(lm.y)
                                    lmx = int(lm.x * 640)
                                    lmy = int(lm.y * 480)

                                    landmarks.append([lmx, lmy])

                                # Drawing landmarks on frames
                                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
                            fore_finger = (landmarks[8][0], landmarks[8][1])
                            center = fore_finger
                            thumb = (landmarks[4][0], landmarks[4][1])
                            cv2.circle(frame, center, 3, (0, 255, 0), -1)
                            print(center[1] - thumb[1])
                            if (thumb[1] - center[1] < 30):
                                bpoints.append(deque(maxlen=512))
                                blue_index += 1
                                gpoints.append(deque(maxlen=512))
                                green_index += 1
                                rpoints.append(deque(maxlen=512))
                                red_index += 1
                                ypoints.append(deque(maxlen=512))
                                yellow_index += 1

                            elif center[1] <= 65:
                                if 40 <= center[0] <= 140:  # Clear Button
                                    bpoints = [deque(maxlen=512)]
                                    gpoints = [deque(maxlen=512)]
                                    rpoints = [deque(maxlen=512)]
                                    ypoints = [deque(maxlen=512)]

                                    blue_index = 0
                                    green_index = 0
                                    red_index = 0
                                    yellow_index = 0

                                    paintWindow[67:, :, :] = 255
                                elif 160 <= center[0] <= 255:
                                    colorIndex = 0  # Blue
                                elif 275 <= center[0] <= 370:
                                    colorIndex = 1  # Green
                                elif 390 <= center[0] <= 485:
                                    colorIndex = 2  # Red
                                elif 505 <= center[0] <= 600:
                                    colorIndex = 3  # Yellow
                            else:
                                if colorIndex == 0:
                                    bpoints[blue_index].appendleft(center)
                                elif colorIndex == 1:
                                    gpoints[green_index].appendleft(center)
                                elif colorIndex == 2:
                                    rpoints[red_index].appendleft(center)
                                elif colorIndex == 3:
                                    ypoints[yellow_index].appendleft(center)
                        # Append the next deques when nothing is detected to avois messing up
                        else:
                            bpoints.append(deque(maxlen=512))
                            blue_index += 1
                            gpoints.append(deque(maxlen=512))
                            green_index += 1
                            rpoints.append(deque(maxlen=512))
                            red_index += 1
                            ypoints.append(deque(maxlen=512))
                            yellow_index += 1

                        # Draw lines of all the colors on the canvas and frame
                        points = [bpoints, gpoints, rpoints, ypoints]
                        # for j in range(len(points[0])):
                        #         for k in range(1, len(points[0][j])):
                        #             if points[0][j][k - 1] is None or points[0][j][k] is None:
                        #                 continue
                        #             cv2.line(paintWindow, points[0][j][k - 1], points[0][j][k], colors[0], 2)
                        for i in range(len(points)):
                            for j in range(len(points[i])):
                                for k in range(1, len(points[i][j])):
                                    if points[i][j][k - 1] is None or points[i][j][k] is None:
                                        continue
                                    cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                                    cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

                        cv2.imshow("Output", frame)
                        cv2.imshow("Paint", paintWindow)

                        if cv2.waitKey(1) == ord('q'):
                            break

                elif curr_tool == "line":
                    xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                    y9 = int(i.landmark[9].y * 480)

                    if index_raised(yi, y9):
                        if not (var_inits):
                            xii, yii = x, y
                            var_inits = True

                        cv2.line(frm, (xii, yii), (x, y), (50, 152, 255), thick)
                    else:
                        if var_inits:
                            cv2.line(mask, (xii, yii), (x, y), 0, thick)
                            var_inits = False

                elif curr_tool == "HandSign":
                    xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                    y9 = int(i.landmark[9].y * 480)

                    if index_raised(yi, y9):
                        if not (var_inits):
                            xii, yii = x, y
                            var_inits = True

                        cv2.rectangle(frm, (xii, yii), (x, y), (0, 255, 255), thick)
                    else:
                        if var_inits:
                            cv2.rectangle(mask, (xii, yii), (x, y), 0, thick)
                            var_inits = False
                elif curr_tool == "circle":
                    xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                    y9 = int(i.landmark[9].y * 480)

                    if index_raised(yi, y9):
                        if not (var_inits):
                            xii, yii = x, y
                            var_inits = True

                        cv2.circle(frm, (xii, yii), int(((xii - x) * 2 + (yii - y)) * 0.5), (255, 255, 0), thick)
                    else:
                        if var_inits:
                            cv2.circle(mask, (xii, yii), int(((xii - x) * 2 + (yii - y)) * 0.5), 0, thick)
                            var_inits = False

                elif curr_tool == "erase":
                    xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                    y9 = int(i.landmark[9].y * 480)

                    if index_raised(yi, y9):
                        cv2.circle(frm, (x, y), 30, (0, 0, 0), -1)
                        cv2.circle(mask, (x, y), 30, 255, -1)

        op = cv2.bitwise_and(frm, frm, mask=mask)
        frm[:, :, 1] = op[:, :, 1]
        frm[:, :, 2] = op[:, :, 2]

        frm[:max_y, ml:max_x] = cv2.addWeighted(tools, 0.7, frm[:max_y, ml:max_x], 0.3, 0)

        cv2.putText(frm, curr_tool, (270 + ml, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Paint App", frm)

        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()





# Create the main tkinter window
root = tk.Tk()
root.title("GUI with Image")

# Load the image using PIL
image = Image.open("C:/Users/Shruti/Downloads/Untitled design.jpg")
image = image.resize((600, 500))  # Resize the image as needed

# Convert the PIL image to a PhotoImage object
photo = ImageTk.PhotoImage(image)

# Create a label to display the image
image_label = tk.Label(root, image=photo)
image_label.pack()

# Create a button
button = tk.Button(root, text="FOR PAINT",command=paint)
button.pack()
button = tk.Button(root, text="FOR HANDSIGN",command=voice)
button.pack()

# Start the tkinter main loop

root.mainloop()