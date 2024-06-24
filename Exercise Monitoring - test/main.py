# Find which exercise is being done
# and call the appropriate function

import cv2
import mediapipe as mp
import numpy as np
from langchain_openai import ChatOpenAI

# Initialize the LLm
llm = ChatOpenAI(openai_api_key='sk-dsdevllmapikey1-2Xm6aVjqITz5BP0nygxoT3BlbkFJfWlZXVOsI7yLo1mfTPHF', 
                 model='gpt-4')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to get the coordinates of a specific landmark
def get_landmark_coords(landmarks, landmark_name):
    landmark = landmarks[mp_pose.PoseLandmark[landmark_name].value]
    return [landmark.x, landmark.y, landmark.z]

# Function to extract and format landmarks from a frame
def extract_landmarks(landmarks):
    formatted_landmarks = {}
    for landmark_name in mp_pose.PoseLandmark:
        formatted_landmarks[landmark_name.name] = get_landmark_coords(landmarks, landmark_name.name)
    return formatted_landmarks

# Function to classify the exercise using GPT
def classify_exercise_with_gpt(landmarks):
    prompt = f"Given the following video, identify which exercise is being performed out of these: {landmarks}. Gait Analysis, Squat, Pushups, Seated Hamstring, Sit to Stand"
    response = llm.invoke(prompt)
    return response.content

# Specify the video file path here
video_file_path = 'files/ex.mp4'

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
            formatted_landmarks = extract_landmarks(landmarks)
            exercise_type = classify_exercise_with_gpt(formatted_landmarks)

            return exercise_type

    cap.release()
    cv2.destroyAllWindows()

# Process the specified video file
print(process_video(video_file_path))
