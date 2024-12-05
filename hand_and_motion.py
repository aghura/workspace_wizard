import cv2
import mediapipe as mp
import numpy as np

def detect_motion_and_hands(video_source=0, motion_threshold=5000):
    """
    Detects motion and hands in a video stream and displays the motion activity with hand landmarks.

    :param video_source: Camera index or path to video file. Default is 0 (webcam).
    :param motion_threshold: Sensitivity threshold for detecting motion.
    """
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Open video source
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return

    print("Press 'q' to quit.")

    # Initialize variables
    prev_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame.")
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        # Detect motion
        motion_detected = False
        motion_display = np.zeros_like(frame)  # Black image for motion display
        if prev_frame is not None:
            # Compute the absolute difference between current frame and previous frame
            frame_diff = cv2.absdiff(prev_frame, gray_frame)
            _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

            # Count non-zero pixels in the thresholded image
            motion_score = np.sum(thresh)
            motion_detected = motion_score > motion_threshold

            if motion_detected:
                print("Motion detected!")

            # Create a visualization of motion
            motion_display = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

            # Overlay hand detection on the motion display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                print("Hands detected!")
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        motion_display,  # The image to draw on
                        hand_landmarks,  # Hand landmarks to draw
                        mp_hands.HAND_CONNECTIONS,  # The connections to draw
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=5),  # Landmarks
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3, circle_radius=2)   # Connections
                    )

        # Update the previous frame
        prev_frame = gray_frame

        # Display the motion detection results with hand landmarks
        cv2.imshow('Motion and Hand Detection', motion_display)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Run motion and hand detection
    detect_motion_and_hands()
