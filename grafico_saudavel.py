from ultralytics import YOLO
import mediapipe as mp
import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt

# Inicializar o modelo YOLO e MediaPipe
model = YOLO('yolov8n.pt')
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Função para calcular ângulo
def calculate_angle(p1, p2, p3):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

# Caminho do vídeo
video_path = r"C:\Users\Pai\Desktop\9periodo\marcha\C_0001_1.MOV"
cap = cv2.VideoCapture(video_path)

# Parâmetros para salvar ângulos em CSV
csv_file = 'angles_data_saudavel.csv'
fieldnames = [
    'frame', 'head_tilt', 'shoulder_angle', 
    'left_knee', 'right_knee', 'left_hip', 'right_hip', 
    'left_ankle', 'right_ankle'
]

with open(csv_file, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Processar o vídeo quadro a quadro
    frame_count = 0
    angles_data = []  # Lista para armazenar os ângulos por frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Usar YOLO para detectar pessoas
        results = model(frame)
        detections = results[0].boxes

        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            cropped_frame = frame[y1:y2, x1:x2]
            cropped_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

            pose_results = pose.process(cropped_rgb)
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark

                # Coordenadas relevantes
                nose = (landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y)
                left_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
                right_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
                left_hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y)
                right_hip = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y)
                left_knee = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y)
                right_knee = (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y)
                left_ankle = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y)
                right_ankle = (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y)

                # Calcular ângulos
                head_tilt_angle = calculate_angle(left_shoulder, nose, right_shoulder)
                shoulder_angle = calculate_angle(left_shoulder, right_shoulder, right_hip)
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
                right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
                left_ankle_angle = calculate_angle(left_knee, left_ankle, (left_ankle[0], left_ankle[1] - 0.1))
                right_ankle_angle = calculate_angle(right_knee, right_ankle, (right_ankle[0], right_ankle[1] - 0.1))

                # Salvar ângulos no CSV
                row = {
                    'frame': frame_count,
                    'head_tilt': head_tilt_angle,
                    'shoulder_angle': shoulder_angle,
                    'left_knee': left_knee_angle,
                    'right_knee': right_knee_angle,
                    'left_hip': left_hip_angle,
                    'right_hip': right_hip_angle,
                    'left_ankle': left_ankle_angle,
                    'right_ankle': right_ankle_angle
                }
                writer.writerow(row)
                angles_data.append([
                    frame_count, head_tilt_angle, shoulder_angle,
                    left_knee_angle, right_knee_angle,
                    left_hip_angle, right_hip_angle,
                    left_ankle_angle, right_ankle_angle
                ])

        frame_count += 1

cap.release()

# Converter dados para numpy para facilitar análise
angles_data = np.array(angles_data)

# Plotar gráficos usando matplotlib
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(angles_data[:, 0], angles_data[:, 1], label="Head Tilt Angle")
plt.title("Head and Shoulder Angles")
plt.ylabel("Angle (°)")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(angles_data[:, 0], angles_data[:, 2], label="Shoulder Angle")
plt.ylabel("Angle (°)")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(angles_data[:, 0], angles_data[:, 3], label="Left Knee")
plt.plot(angles_data[:, 0], angles_data[:, 4], label="Right Knee")
plt.title("Knee Angles")
plt.ylabel("Angle (°)")
plt.legend()

plt.xlabel("Frame")
plt.tight_layout()
plt.show()
