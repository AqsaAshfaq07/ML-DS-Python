import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Function to get the coordinates of a specific landmark
def get_landmark_coords(landmarks, landmark_name):
    landmark = landmarks[mp_pose.PoseLandmark[landmark_name].value]
    return [landmark.x, landmark.y]

# Function to check the current position (Top or Bottom)
def check_position(landmarks):
    shoulder = get_landmark_coords(landmarks, 'LEFT_SHOULDER')
    elbow = get_landmark_coords(landmarks, 'LEFT_ELBOW')
    wrist = get_landmark_coords(landmarks, 'LEFT_WRIST')
    hip = get_landmark_coords(landmarks, 'LEFT_HIP')

    shoulder_elbow_angle = calculate_angle(shoulder, elbow, wrist)
    elbow_wrist_angle = calculate_angle(elbow, wrist, shoulder)
    wrist_hip_distance = np.linalg.norm(np.array(wrist) - np.array(hip))

    if shoulder_elbow_angle < 120 and elbow_wrist_angle > 160 and wrist_hip_distance > 0.4:
        return "Bottom"
    elif shoulder_elbow_angle > 160 and elbow_wrist_angle < 120:
        return "Top"
    else:
        return "Transition"

# Function to evaluate bottom position
def evaluate_bottom_position(landmarks):
    shoulder = get_landmark_coords(landmarks, 'LEFT_SHOULDER')
    elbow = get_landmark_coords(landmarks, 'LEFT_ELBOW')
    wrist = get_landmark_coords(landmarks, 'LEFT_WRIST')

    shoulder_elbow_angle = calculate_angle(shoulder, elbow, wrist)
    elbow_wrist_angle = calculate_angle(elbow, wrist, shoulder)

    feedback = "Good posture"
    if shoulder_elbow_angle > 120:
        feedback = "Keep your elbows closer"
    elif elbow_wrist_angle < 160:
        feedback = "Extend your arms fully"

    return feedback

# Function to evaluate top position
def evaluate_top_position(landmarks):
    shoulder = get_landmark_coords(landmarks, 'LEFT_SHOULDER')
    elbow = get_landmark_coords(landmarks, 'LEFT_ELBOW')
    wrist = get_landmark_coords(landmarks, 'LEFT_WRIST')

    shoulder_elbow_angle = calculate_angle(shoulder, elbow, wrist)

    feedback = "Good posture"
    if shoulder_elbow_angle < 160:
        feedback = "Keep your arms extended"

    return feedback

# Specify the video file path here
video_file_path = 'files/pushups.mp4'

# Main function to process the video file
def process_video(file_path):
    cap = cv2.VideoCapture(file_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            position = check_position(landmarks)
            feedback = ""  # Initialize feedback variable

            if position == "Bottom":
                feedback = evaluate_bottom_position(landmarks)
            elif position == "Top":
                feedback = evaluate_top_position(landmarks)

            cv2.putText(frame, f'Position: {position}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, feedback, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Push-up Feedback', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Process the specified video file
process_video(video_file_path)
