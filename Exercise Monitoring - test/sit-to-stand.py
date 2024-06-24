import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

# Constants for angle thresholds
STANDING_HIP_ANGLE = 160
STANDING_KNEE_ANGLE = 160
STANDING_ANKLE_ANGLE_LOWER = 80
STANDING_ANKLE_ANGLE_UPPER = 100

SEATED_HIP_ANGLE = 90
SEATED_KNEE_ANGLE = 90

def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def get_landmark_coords(landmarks, landmark_name):
    landmark = landmarks[mp_pose.PoseLandmark[landmark_name].value]
    return [landmark.x, landmark.y]

def check_position(landmarks):
    hip = get_landmark_coords(landmarks, 'LEFT_HIP')
    knee = get_landmark_coords(landmarks, 'LEFT_KNEE')
    ankle = get_landmark_coords(landmarks, 'LEFT_ANKLE')
    shoulder = get_landmark_coords(landmarks, 'LEFT_SHOULDER')

    hip_angle = calculate_angle(shoulder, hip, knee)
    knee_angle = calculate_angle(hip, knee, ankle)
    ankle_angle = calculate_angle(knee, ankle, (ankle[0], ankle[1] - 1))  # Approximate vertical line

    if hip_angle > STANDING_HIP_ANGLE and knee_angle > STANDING_KNEE_ANGLE and STANDING_ANKLE_ANGLE_LOWER < ankle_angle < STANDING_ANKLE_ANGLE_UPPER:
        return "Standing"
    elif hip_angle < SEATED_HIP_ANGLE and knee_angle < SEATED_KNEE_ANGLE:
        return "Seated"
    else:
        return "Transition"

def evaluate_standing_position(landmarks):
    hip = get_landmark_coords(landmarks, 'LEFT_HIP')
    knee = get_landmark_coords(landmarks, 'LEFT_KNEE')
    ankle = get_landmark_coords(landmarks, 'LEFT_ANKLE')

    knee_angle = calculate_angle(hip, knee, ankle)

    feedback = "Good standing posture"
    if knee_angle < 170:
        feedback = "Straighten your knees"
    elif knee_angle > 190:
        feedback = "Relax your knees"

    return feedback, knee_angle

def evaluate_seated_position(landmarks):
    hip = get_landmark_coords(landmarks, 'LEFT_HIP')
    knee = get_landmark_coords(landmarks, 'LEFT_KNEE')
    ankle = get_landmark_coords(landmarks, 'LEFT_ANKLE')

    hip_angle = calculate_angle(get_landmark_coords(landmarks, 'LEFT_SHOULDER'), hip, knee)
    knee_angle = calculate_angle(hip, knee, ankle)

    feedback = "Good seated posture"
    if hip_angle < 80:
        feedback = "Lean back slightly"
    elif hip_angle > 100:
        feedback = "Sit up straight"

    if knee_angle < 80:
        feedback += " and bend your knees more"
    elif knee_angle > 100:
        feedback += " and straighten your knees"

    return feedback, hip_angle, knee_angle

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
            feedback, knee_angle = evaluate_standing_position(landmarks)
            cv2.putText(frame, f'Knee Angle: {knee_angle:.2f}', 
                        tuple(np.multiply(get_landmark_coords(landmarks, 'LEFT_KNEE'), [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        elif position == "Seated":
            feedback, hip_angle, knee_angle = evaluate_seated_position(landmarks)
            cv2.putText(frame, f'Hip Angle: {hip_angle:.2f}', 
                        tuple(np.multiply(get_landmark_coords(landmarks, 'LEFT_HIP'), [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Knee Angle: {knee_angle:.2f}', 
                        tuple(np.multiply(get_landmark_coords(landmarks, 'LEFT_KNEE'), [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, feedback, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Sit-to-Stand Feedback', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
