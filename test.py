import cv2
import mediapipe as mp

# Initialize MediaPipe pose detector
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Check if camera opened correctly
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

paused = False   # This variable will control pause/unpause

while True:
    # Only read new frame if not paused
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("❌ Can't receive frame. Exiting ...")
            break

        # Convert frame to RGB (for MediaPipe)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # Draw pose landmarks (if detected)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

    # Show the (possibly paused) frame
    cv2.imshow('Pose Detection', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break   # Quit when 'q' pressed
    elif key == ord('p'):
        paused = not paused   # Toggle pause/unpause

cap.release()
cv2.destroyAllWindows()
