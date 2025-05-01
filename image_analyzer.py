# Import necessary libraries
import mediapipe as mp
import cv2

# Initialize MediaPipe's Pose module
mp_pose = mp.solutions.pose

# Initialize drawing utilities to visualize landmarks (optional)
mp_drawing = mp.solutions.drawing_utils

# Function to process an image and print body part positions
def extract_and_print_landmarks(image_path):
    # Initialize pose estimator
    with mp_pose.Pose(static_image_mode=True) as pose:
        # Load the image from the given path
        image = cv2.imread(image_path)

        # Check if image loaded successfully
        if image is None:
            print("Error: Could not load image. Please check the path.")
            return

        # Convert BGR image (OpenCV format) to RGB (MediaPipe expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image to detect pose landmarks
        results = pose.process(image_rgb)

        # Check if landmarks are detected
        if not results.pose_landmarks:
            print("No pose landmarks detected in the image.")
            return

        # Get the list of landmarks
        landmarks = results.pose_landmarks.landmark

        # List of body parts names (corresponds to landmark indices)
        body_parts = mp_pose.PoseLandmark

        # Print each body part name and its (x, y, z) coordinates
        print("\nDetected body landmarks:\n")
        for part in body_parts:
            landmark = landmarks[part.value]
            print(f"{part.name}: x={landmark.x:.3f}, y={landmark.y:.3f}, z={landmark.z:.3f}")

        # OPTIONAL: Draw the landmarks on the image and show it
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
        # Display the image with landmarks (optional visualization)
        cv2.imshow('Pose Landmarks', annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# ======== USAGE ========
# Replace this with your actual image path
image_path = 'images/IMG_2882 3.PNG'
extract_and_print_landmarks(image_path)
