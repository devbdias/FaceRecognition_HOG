import csv
import face_recognition
import os

import datetime

inicio = datetime.datetime.now()

known_faces_dir = r'C:\Users\bruno\OneDrive\Área de Trabalho\FaceRecognition_v2_HOG\src\assets\conhecidos'
unknown_faces_dir = r'C:\Users\bruno\OneDrive\Área de Trabalho\FaceRecognition_v2_HOG\src\assets\desconhecidos'

# Load known faces
known_faces = []
known_names = []
for filename in os.listdir(known_faces_dir):
    image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
    face_locations = face_recognition.face_locations(image)
    if face_locations:
        face_encoding = face_recognition.face_encodings(image, face_locations)[0]
        known_faces.append(face_encoding)
        known_names.append(os.path.splitext(filename)[0])

# Load unknown faces
unknown_faces = []
unknown_names = []
for filename in os.listdir(unknown_faces_dir):
    image = face_recognition.load_image_file(os.path.join(unknown_faces_dir, filename))
    face_locations = face_recognition.face_locations(image)
    if face_locations:
        face_encoding = face_recognition.face_encodings(image, face_locations)[0]
        unknown_faces.append(face_encoding)
        unknown_names.append(os.path.splitext(filename)[0])

# Recognize faces
with open(r'src\results\accuracy_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Unknown Person', 'Known Person', 'Match', 'Accuracy'])
    for i, unknown_face_encoding in enumerate(unknown_faces):
        for j, known_face_encoding in enumerate(known_faces):
            results = face_recognition.compare_faces([known_face_encoding], unknown_face_encoding)
            match = True in results
            accuracy = results.count(True) / len(results) * 100
            writer.writerow([unknown_names[i], known_names[j], match, accuracy])

            print(f"Comparing {unknown_names[i]} with {known_names[j]}:")
            if match:
                print(f"  {known_names[j]}: match")
            else:
                print(f"  {known_names[j]}: no match")
            print(f"  Accuracy: {accuracy:.2f}%")

fim = datetime.datetime.now()

print(f'Tempo de execção: {fim - inicio}')