from ultralytics import YOLO
import mediapipe as mp
import cv2
import numpy as np

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
    angle = np.arccos(cos_angle)
    return np.degrees(angle)

video_path = r"C:\Users\Pai\Desktop\9periodo\marcha\C_0001_1.MOV"  # Substitua pelo caminho do seu vídeo
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec de vídeo (pode variar dependendo do sistema)
fps = cap.get(cv2.CAP_PROP_FPS)  # Manter a mesma taxa de quadros do vídeo original
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Criar o objeto VideoWriter
out = cv2.VideoWriter('output_saudavel.avi', fourcc, fps, (width, height)) # Salvar como 'output.avi'

# Processar o vídeo quadro a quadro
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Usar YOLO para detectar pessoas
    results = model(frame)
    detections = results[0].boxes  # Caixas detectadas

    for detection in detections:
        # Extrair coordenadas da caixa delimitadora
        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        cropped_frame = frame[y1:y2, x1:x2]  # Recortar a região da pessoa

        # Converter para RGB (necessário para MediaPipe)
        cropped_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

        # Usar MediaPipe para estimativa de pose
        pose_results = pose.process(cropped_rgb)

        if pose_results.pose_landmarks:
            # Desenhar os pontos-chave na região recortada
            mp.solutions.drawing_utils.draw_landmarks(
                cropped_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

            # Extração de coordenadas das articulações relevantes
            landmarks = pose_results.pose_landmarks.landmark

            # Coordenadas da cabeça e tronco
            nose = (landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y)
            left_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
            right_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
            left_hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y)
            right_hip = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y)

            # Coordenadas dos membros inferiores
            left_knee = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y)
            right_knee = (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y)
            left_ankle = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y)
            right_ankle = (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y)

            # Calcular ângulos do tronco e cabeça
            shoulder_angle = calculate_angle(left_shoulder, right_shoulder, right_hip)  # Ângulo entre ombro e quadril
            head_tilt_angle = calculate_angle(left_shoulder, nose, right_shoulder)  # Inclinação da cabeça

            # Calcular ângulos dos membros inferiores
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)  # Ângulo do joelho esquerdo
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)  # Ângulo do joelho direito
            left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)  # Ângulo do quadril esquerdo
            right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)  # Ângulo do quadril direito
            left_ankle_angle = calculate_angle(left_knee, left_ankle, (left_ankle[0], left_ankle[1] - 0.1))  # Ângulo do tornozelo esquerdo (estimado)
            right_ankle_angle = calculate_angle(right_knee, right_ankle, (right_ankle[0], right_ankle[1] - 0.1))  # Ângulo do tornozelo direito (estimado)

             # Adicionar informações ao frame
            info_text = (
                f"Ângulos:\n"
                f"Cabeça: {head_tilt_angle:.2f}°\n"
                f"Ombros: {shoulder_angle:.2f}°\n"
                f"Joelho Esq: {left_knee_angle:.2f}°\n"
                f"Joelho Dir: {right_knee_angle:.2f}°\n"
                f"Quadril Esq: {left_hip_angle:.2f}°\n"
                f"Quadril Dir: {right_hip_angle:.2f}°\n"
                f"Tornozelo Esq: {left_ankle_angle:.2f}°\n"
                f"Tornozelo Dir: {right_ankle_angle:.2f}°"
            )
            y0, dy = 20, 20
            for i, line in enumerate(info_text.split('\n')):
                y = y0 + i * dy
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Substituir a região original pela recortada com landmarks
        frame[y1:y2, x1:x2] = cropped_frame

    # Escrever o frame processado no arquivo de saída
    out.write(frame)

    # Exibir o resultado (usando cv2_imshow para Colab)
    cv2.imshow("Processed Frame",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
#cv2.destroyAllWindows()

