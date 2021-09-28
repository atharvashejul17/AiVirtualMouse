import mediapipe as mp #Mediapipe offers open source cross-platform, customizble ML solutions for live and streaming media.
import cv2
import numpy as np
import uuid #Universal Unique Identifier, is a python library which helps in generating random objects of 128 bits as ids. It provides the uniqueness as it generates ids on the basis of time, Computer hardware (MAC etc.). Useful in generating random documents, addresses etc.
import os

mp_drawing = mp.solutions.drawing_utils  # component of mediapipe. Used to drawing utility and rendering the landmarks
mp_hands = mp.solutions.hands  # component of mediapipe. This will get the hand model for us which is already thier in mediapipe.

cap = cv2.VideoCapture(0)  # Component of openCV. To turn on the web cam

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()  # In frave var we will read the results of webcam

        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip on horizontal bcoz webcam take left hand as right n vise versa
        image = cv2.flip(image, 1)

        # Set flag to stop copying any other image
        image.flags.writeable = False

        #Process and Detect the hand
        results = hands.process(image)

        # Once hand is detected Set flag to true
        image.flags.writeable = True

        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        print(results)

        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):  # .HAND_CONNECTIONS is lines which connects your one joint to another
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, #mp_drawing command used to draw land marks on coordinates
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),  #For landmarks dots
                                          mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                          )

        cv2.imshow('Hand Tracking', image)  # Opens new window which shows webcam window

        if cv2.waitKey(10) & 0xFF == ord('q'):  # To stop the window
            break

cap.release()  #Stoping Webcam function
cv2.destroyAllWindows()  #Closing all windows which are open with that command
mp_drawing.DrawingSpec()

os.mkdir('Output Images')
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip on horizontal
        image = cv2.flip(image, 1)

        # Set flag
        image.flags.writeable = False

        # Detections
        results = hands.process(image)

        # Set flag to true
        image.flags.writeable = True

        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Detections
        print(results)
        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                          )

        # Save our image
        cv2.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), image) #Only this part is add to save image . Other than this everything is same
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()