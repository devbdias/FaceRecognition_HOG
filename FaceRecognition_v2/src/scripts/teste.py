import face_recognition as fr
import os
import pandas as pd
import numpy as np
import cv2

# Verifica se a GPU está disponível
if not cv2.cuda.getCudaEnabledDeviceCount():
    print("Não há GPU disponível.")
    exit()

# Configura a captura de vídeo
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Carrega as imagens de treinamento e seus respectivos nomes
known_faces_dir = "C:\Users\bruno\Downloads\FaceRecognition_v2\FaceRecognition_v2\src\assets\conhecidos"
known_faces = []
known_names = []
for filename in os.listdir(known_faces_dir):
    image = fr.load_image_file(known_faces_dir + filename)
    face_encoding = fr.face_encodings(image)[0]
    known_faces.append(face_encoding)
    known_names.append(os.path.splitext(filename)[0])

# Loop principal
while True:
    # Captura um quadro de vídeo
    ret, frame = video_capture.read()

    # Reduz o tamanho do quadro para acelerar o processamento
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Converte o quadro de BGR para RGB
    rgb_small_frame = small_frame[:, :, ::-1]

    # Localiza os rostos no quadro
    face_locations = fr.face_locations(rgb_small_frame, model="cnn")
    face_encodings = fr.face_encodings(rgb_small_frame, face_locations)

    # Para cada rosto encontrado, tenta reconhecê-lo
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = fr.compare_faces(known_faces, face_encoding)
        name = "Desconhecido"

        # Encontra a face mais próxima
        face_distances = fr.face_distance(known_faces, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_names[best_match_index]

        # Desenha um retângulo e um rótulo com o nome do rosto
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Exibe o quadro resultante
    cv2.imshow('Video', frame)

    # Interrompe o loop quando a tecla 'q' é pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
video_capture.release()
cv2.destroyAllWindows()
