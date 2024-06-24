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

# Function to check the current gait phase based on joint angles
def check_gait_phase(landmarks):
    hip = get_landmark_coords(landmarks, 'LEFT_HIP')
    knee = get_landmark_coords(landmarks, 'LEFT_KNEE')
    ankle = get_landmark_coords(landmarks, 'LEFT_ANKLE')
    shoulder = get_landmark_coords(landmarks, 'LEFT_SHOULDER')

    hip_angle = calculate_angle(shoulder, hip, knee)
    knee_angle = calculate_angle(hip, knee, ankle)

    if hip_angle < 140 and knee_angle > 170:
        return "Heel Strike"
    elif hip_angle > 170 and knee_angle > 160:
        return "Mid Stance"
    elif hip_angle > 160 and knee_angle < 150:
        return "Toe Off"
    else:
        return "Transition"

# Function to evaluate the gait phase and provide feedback
def evaluate_gait_phase(landmarks, phase):
    hip = get_landmark_coords(landmarks, 'LEFT_HIP')
    knee = get_landmark_coords(landmarks, 'LEFT_KNEE')
    ankle = get_landmark_coords(landmarks, 'LEFT_ANKLE')

    hip_angle = calculate_angle(get_landmark_coords(landmarks, 'LEFT_SHOULDER'), hip, knee)
    knee_angle = calculate_angle(hip, knee, ankle)

    feedback = "Good posture"

    if phase == "Heel Strike":
        if hip_angle > 140:
            feedback = "Lower your hip slightly"
        if knee_angle < 170:
            feedback = "Extend your knee more"
    elif phase == "Mid Stance":
        if hip_angle < 170:
            feedback = "Extend your hip more"
        if knee_angle < 160:
            feedback = "Bend your knee slightly"
    elif phase == "Toe Off":
        if hip_angle < 160:
            feedback = "Raise your hip slightly"
        if knee_angle > 150:
            feedback = "Bend your knee more"

    if "Lower your hip" not in feedback and "Extend your knee" not in feedback and "Bend your knee" not in feedback:
        feedback = "Good posture"

    return feedback

# Specify the video file path here
video_file_path = 'files/gait.mp4'

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
            phase = check_gait_phase(landmarks)
            feedback = evaluate_gait_phase(landmarks, phase)

            cv2.putText(frame, f'Phase: {phase}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, feedback, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Gait Training Feedback', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Process the specified video file
process_video(video_file_path)
