from ultralytics import YOLO
import mediapipe as mp
import cv2
import numpy as np

def calculate_angle(p1, p2, p3):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)

    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cos_angle)
    return np.degrees(angle)

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# YOLO and MediaPipe setup
model = YOLO('yolov8n.pt')
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

video_path = r"C:\\Users\\Pai\\Desktop\\9periodo\\marcha\\P_0001_1.MOV"
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

total_distance = 0
time_per_frame = 1 / fps
step_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results[0].boxes

    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        cropped_frame = frame[y1:y2, x1:x2]
        cropped_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

        pose_results = pose.process(cropped_rgb)

        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            left_ankle = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y)
            right_ankle = (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, 
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y)

            # Calculate spatial parameters
            step_length = calculate_distance(left_ankle, right_ankle)
            total_distance += step_length

            # Calculate angular parameters
            left_knee = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y)
            right_knee = (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, 
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y)
            left_hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y)
            right_hip = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y)

            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

            # Display information
            info_text = (f"Distância Percorrida: {total_distance:.2f} m\n"
                         f"Comprimento do Passo: {step_length:.2f} m\n"
                         f"Ângulo Joelho Esq: {left_knee_angle:.2f}°\n"
                         f"Ângulo Joelho Dir: {right_knee_angle:.2f}°")

            y0, dy = 20, 20
            for i, line in enumerate(info_text.split('\n')):
                y = y0 + i * dy
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    out.write(frame)
    cv2.imshow("Processed Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
