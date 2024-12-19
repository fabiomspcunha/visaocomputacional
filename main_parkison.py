import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import mediapipe as mp

# Inicializando o modelo YOLO e o MediaPipe Pose
model = YOLO('yolov8n.pt')
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(p1, p2, p3):
    """
    Calcula o ângulo entre três pontos.
    :param p1: Coordenadas (x, y) do primeiro ponto.
    :param p2: Coordenadas (x, y) do vértice.
    :param p3: Coordenadas (x, y) do terceiro ponto.
    :return: Ângulo em graus.
    """
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)

    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_stride_length(previous_foot, current_foot):
    """
    Calcula o comprimento da passada com base nas coordenadas do pé em dois momentos consecutivos.
    :param previous_foot: Coordenadas (x, y) do pé no momento anterior.
    :param current_foot: Coordenadas (x, y) do pé no momento atual.
    :return: Comprimento da passada em metros.
    """
    return np.linalg.norm(np.array(previous_foot) - np.array(current_foot))

video_path = r"C:\Users\Pai\Desktop\9periodo\marcha\P_0001_1.MOV"
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

total_distance = 0

# Variáveis para armazenar os resultados
timestamps = []
distances = []
step_lengths = []
left_knee_angles = []
right_knee_angles = []
stride_lengths = []

previous_left_foot = None
previous_right_foot = None

time = 0
frame_time = 1 / fps

def extract_coordinates(landmark):
    return (landmark.x, landmark.y)

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
            mp.solutions.drawing_utils.draw_landmarks(
                cropped_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

            landmarks = pose_results.pose_landmarks.landmark
            left_ankle = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
                            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y)
            right_ankle = (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, 
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y)

            left_foot = extract_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
            right_foot = extract_coordinates(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
            left_knee = extract_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
            right_knee = extract_coordinates(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
            left_hip = extract_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
            right_hip = extract_coordinates(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])

            left_knee_angle = calculate_angle(left_hip, left_knee, left_foot)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_foot)
            
            step_length = calculate_distance(left_ankle, right_ankle)
            total_distance += step_length

            if previous_left_foot is not None and previous_right_foot is not None:
                stride_length = calculate_stride_length(previous_left_foot, left_foot)
            else:
                stride_length = 0

            previous_left_foot = left_foot
            previous_right_foot = right_foot

            timestamps.append(time)
            distances.append(total_distance)  # Implementar cálculo de distância percorrida
            step_lengths.append(step_length)  # Implementar cálculo de comprimento do passo
            left_knee_angles.append(left_knee_angle)
            right_knee_angles.append(right_knee_angle)
            stride_lengths.append(stride_length)

            time += frame_time

            info_text = (
                f"Ângulo Joelho Esq: {left_knee_angle:.2f}°\n"
                f"Ângulo Joelho Dir: {right_knee_angle:.2f}°\n"
                f"Comprimento do Passo: {step_length:.2f} m\n"
                f"Comprimento da Passada: {stride_length:.2f}m"
            )
            y0, dy = 20, 20
            for i, line in enumerate(info_text.split('\n')):
                y = y0 + i * dy
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        frame[y1:y2, x1:x2] = cropped_frame

    out.write(frame)
    cv2.imshow("Processed Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Salvando os resultados em um arquivo CSV
with open('analysis_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Tempo (s)", "Distância Percorrida (m)", "Comprimento do Passo (m)", "Ângulo Joelho Esq (°)", "Ângulo Joelho Dir (°)", "Comprimento da Passada (m)"])
    writer.writerows(zip(timestamps, distances, step_lengths, left_knee_angles, right_knee_angles, stride_lengths))

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(timestamps, left_knee_angles, label="Ângulo Joelho Esq (°)")
plt.plot(timestamps, right_knee_angles, label="Ângulo Joelho Dir (°)")
plt.xlabel("Tempo (s)")
plt.ylabel("Medições")
plt.title("Parâmetros da Marcha ao Longo do Tempo")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(timestamps, stride_lengths, label="Comprimento da Passada (m)")
plt.xlabel("Tempo (s)")
plt.ylabel("Medições")
plt.title("Parâmetros da Marcha ao Longo do Tempo")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(timestamps, step_lengths, label="Comprimento do Passo (m)", color="red")
plt.xlabel("Tempo (s)")
plt.ylabel("Medições")
plt.title("Parâmetros da Marcha ao Longo do Tempo")
plt.legend()
plt.grid()
plt.show()


