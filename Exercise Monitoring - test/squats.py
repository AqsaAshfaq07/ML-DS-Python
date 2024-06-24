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

# Function to check the current position (Standing or Lowered)
def check_position(landmarks):
    hip = get_landmark_coords(landmarks, 'LEFT_HIP')
    knee = get_landmark_coords(landmarks, 'LEFT_KNEE')
    ankle = get_landmark_coords(landmarks, 'LEFT_ANKLE')
    shoulder = get_landmark_coords(landmarks, 'LEFT_SHOULDER')

    hip_angle = calculate_angle(shoulder, hip, knee)
    knee_angle = calculate_angle(hip, knee, ankle)
    ankle_angle = calculate_angle(knee, ankle, (ankle[0], ankle[1] - 1))  # Approximate vertical line

    if hip_angle > 160 and knee_angle > 160 and 80 < ankle_angle < 100:
        return "Standing"
    elif hip_angle < 120 and knee_angle < 120:
        return "Lowered"
    else:
        return "Transition"

# Function to evaluate standing position
def evaluate_standing_position(landmarks):
    hip = get_landmark_coords(landmarks, 'LEFT_HIP')
    knee = get_landmark_coords(landmarks, 'LEFT_KNEE')
    ankle = get_landmark_coords(landmarks, 'LEFT_ANKLE')

    knee_angle = calculate_angle(hip, knee, ankle)

    feedback = "Good posture"
    if knee_angle < 170:
        feedback = "Straighten your knees"
    elif knee_angle > 190:
        feedback = "Relax your knees"

    return feedback

# Function to evaluate lowered position
def evaluate_lowered_position(landmarks):
    hip = get_landmark_coords(landmarks, 'LEFT_HIP')
    knee = get_landmark_coords(landmarks, 'LEFT_KNEE')
    ankle = get_landmark_coords(landmarks, 'LEFT_ANKLE')
    shoulder = get_landmark_coords(landmarks, 'LEFT_SHOULDER')

    hip_angle = calculate_angle(shoulder, hip, knee)
    knee_angle = calculate_angle(hip, knee, ankle)

    feedback = "Good posture"
    if hip_angle < 70:
        feedback = "Lower your hips more"
    elif hip_angle > 120:
        feedback = "Raise your hips a bit"

    if knee_angle < 70:
        feedback += " and bend your knees more"
    elif knee_angle > 120:
        feedback += " and bend your knees less"

    return feedback

# Specify the video file path here
video_file_path = 'files/squatt.mp4'

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

            if position == "Standing":
                feedback = evaluate_standing_position(landmarks)
            elif position == "Lowered":
                feedback = evaluate_lowered_position(landmarks)

            cv2.putText(frame, f'Position: {position}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, feedback, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Exercise Feedback', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Process the specified video file
process_video(video_file_path)
