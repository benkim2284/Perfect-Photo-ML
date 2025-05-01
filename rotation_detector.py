import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to calculate angle between two points
def calculate_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = (1-p2[1]) - (1-p1[1])

    angle = np.degrees(np.arctan2(dy, dx))
    return angle

# Load image
image = cv2.imread('images/test_photo.jpg')  # Path to your image

# Convert to RGB (MediaPipe expects RGB)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize Pose model
with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
    # Process image to find pose landmarks
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # Extract shoulder landmarks
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        print("left_shoulder")
        print(left_shoulder)
        print()
        print("right_shoulder")
        print(right_shoulder)

        # Get coordinates of shoulders
        left_shoulder_coords = (left_shoulder.x * image.shape[1], left_shoulder.y * image.shape[0])
        right_shoulder_coords = (right_shoulder.x * image.shape[1], right_shoulder.y * image.shape[0])

        # Calculate the angle of shoulder line
        shoulder_angle = calculate_angle(right_shoulder_coords, left_shoulder_coords)

        # Determine if the person is rotated
        rotation_threshold = 3 # Degrees (tweak as needed)
        print(f'angle is {shoulder_angle}')
        if abs(shoulder_angle) < rotation_threshold:
            print("Person is facing straight (not rotated).")
        else:
            if shoulder_angle > 0:
                print("Person is rotated to the right.")
            else:
                print("Person is rotated to the left.")

        # OPTIONAL: Draw shoulder line on the image for visualization
        cv2.line(image, 
                 (int(left_shoulder_coords[0]), int(left_shoulder_coords[1])), 
                 (int(right_shoulder_coords[0]), int(right_shoulder_coords[1])), 
                 (0, 255, 0), 2)

        # Show the image with the shoulder line
        cv2.imshow('Pose Detection', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("No pose landmarks found in the image.")
