import face_recognition
import cv2
import numpy as np
import os

# Configuração da webcam
video_capture = cv2.VideoCapture(0)

# Caminho da pasta com as imagens conhecidas
known_faces_dir = r'C:\Users\bruno\OneDrive\Área de Trabalho\FaceRecognition_v2_HOG\src\assets\conhecidos'

# Lista de rostos conhecidos e seus nomes
known_faces = []
known_names = []

# Carrega as imagens conhecidas e seus nomes
for filename in os.listdir(known_faces_dir):
    image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
    face_encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(face_encoding)
    known_names.append(os.path.splitext(filename)[0])

# Loop de captura da webcam
while True:
    # Captura um frame da webcam
    ret, frame = video_capture.read()

    # Converte o frame para RGB (é necessário para o face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Encontra os rostos no frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop pelos rostos encontrados
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compara o rosto encontrado com os rostos conhecidos
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        # Verifica se houve uma correspondência
        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]
        
        # Calcula e exibe a acurácia em tempo real
        accuracy = (matches.count(True) / len(matches)) * (matches.count(True) * len(matches)) * 100

        if accuracy > 0:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        # Desenha o nome e a caixa ao redor do rosto
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Adiciona o nome abaixo do quadro
        cv2.putText(frame, name, (left + 6, bottom + 25), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
        
        # Adiciona a acurácia abaixo do nome
        cv2.putText(frame, f"Accuracy: {accuracy:.2f}%", (left + 6, bottom + 45), cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1)

    # Redimensiona o frame para a largura da janela
    height, width, _ = frame.shape
    window_width = 720
    scale = window_width / width
    window_height = 600#int(scale * height)
    frame = cv2.resize(frame, (window_width, window_height))

    # Exibe o resultado
    cv2.imshow('Video', frame)

    # Fecha a janela quando a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a webcam e fecha as janelas
video_capture.release()
cv2.destroyAllWindows()
