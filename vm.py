import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize MediaPipe and pyautogui
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

# Time variables
prev_click_time = 0
clicking = False
dragging = False
drag_start_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark

            # Landmarks
            index_finger = landmarks[8]
            thumb = landmarks[4]
            pinky = landmarks[20]

            # Get position
            x = int(index_finger.x * w)
            y = int(index_finger.y * h)
            screen_x = np.interp(x, (0, w), (0, screen_width))
            screen_y = np.interp(y, (0, h), (0, screen_height))
            pyautogui.moveTo(screen_x, screen_y, duration=0.1)

            # Pinch distance (Index & Thumb)
            pinch_dist = np.linalg.norm(np.array([index_finger.x, index_finger.y]) - np.array([thumb.x, thumb.y])) * w

            # Pinky distance (for right click)
            pinky_dist = np.linalg.norm(np.array([pinky.x, pinky.y]) - np.array([thumb.x, thumb.y])) * w

            current_time = time.time()

            # Left Click
            if pinch_dist < 30:
                if not clicking:
                    if current_time - prev_click_time < 0.3:
                        pyautogui.doubleClick()
                        cv2.putText(frame, "Double Click", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    else:
                        pyautogui.click()
                        cv2.putText(frame, "Click", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    prev_click_time = current_time
                    clicking = True

                    # Start drag timer
                    drag_start_time = current_time
                else:
                    # Dragging logic
                    if not dragging and current_time - drag_start_time > 1:
                        dragging = True
                        pyautogui.mouseDown()
                        cv2.putText(frame, "Dragging", (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False
                clicking = False

            # Right Click
            if pinky_dist < 30:
                pyautogui.rightClick()
                cv2.putText(frame, "Right Click", (x, y - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                time.sleep(1)  # Avoid multiple right-clicks in a row

    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
