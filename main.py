# Import necessary libraries
import mediapipe as mp
import cv2

# Initialize MediaPipe's Pose module and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Start webcam video capture
cap = cv2.VideoCapture(0)  # 0 = default camera

# Initialize pose estimator (live mode: static_image_mode=False)
with mp_pose.Pose(static_image_mode=False,
                  model_complexity=1,  # Can adjust 0 (fast) to 2 (accurate)
                  min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        # Flip image horizontally (mirror view, like a selfie)
        # frame = cv2.flip(frame, 1)

        # Convert BGR frame (OpenCV format) to RGB (MediaPipe expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect pose landmarks
        results = pose.process(frame_rgb)


        # If landmarks are detected, process them
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Example: Print LEFT_SHOULDER coordinates live
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

            print(f"LEFT_SHOULDER: x={right_shoulder.x:.3f}, y={right_shoulder.y:.3f}, z={right_shoulder.z:.3f}")
            print(f"RIGHT_SHOULDER: x={left_shoulder.x:.3f}, y={left_shoulder.y:.3f}, z={left_shoulder.z:.3f}")

            # OPTIONAL: Draw pose landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            # Loop through each landmark and label it
            h, w, _ = frame.shape  # Get frame dimensions
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                cx, cy = int(landmark.x * w), int(landmark.y * h)

                # Get landmark name using the Enum
                landmark_name = mp_pose.PoseLandmark(idx).name

                # Draw the text label next to the point
                cv2.putText(frame, landmark_name, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

        # Display the video frame
        cv2.imshow('Live Pose Detection', frame)

        # Press 'q' key to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(str(results.pose_landmarks.landmark))
            for landmark in mp_pose.PoseLandmark:
                print(f"Name: {landmark.name}, Index: {landmark.value}")
            break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
