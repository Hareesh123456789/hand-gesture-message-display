import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load model
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

def fingers_up(landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    fingers.append(1 if landmarks[tips[0]].x < landmarks[tips[0]-1].x else 0)

    # Other fingers
    for i in range(1, 5):
        fingers.append(1 if landmarks[tips[i]].y < landmarks[tips[i]-2].y else 0)

    return fingers

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)
    message = "Show Gesture"

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        pattern = fingers_up(hand)

        if pattern == [1,1,1,1,1]:
            message = "HELLO"
        elif pattern == [0,0,0,0,0]:
            message = "STOP"
        elif pattern == [1,0,0,0,0]:
            message = "YES"
        elif pattern == [0,1,0,0,0]:
            message = "NO"
        elif pattern == [0,1,1,0,0]:
            message = "THANK YOU"
        else:
            message = "UNKNOWN"

        # Draw landmarks
        for lm in hand:
            x = int(lm.x * frame.shape[1])
            y = int(lm.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0,255,0), -1)

    cv2.rectangle(frame, (0,0), (450,80), (0,0,0), -1)
    cv2.putText(frame, message, (20,60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)

    cv2.imshow("Hand Gesture Message Display", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
