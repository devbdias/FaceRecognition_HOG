# %%
import csv
import face_recognition
import os
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

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
results = []
for i, unknown_face_encoding in enumerate(unknown_faces):
    for j, known_face_encoding in enumerate(known_faces):
        match = face_recognition.compare_faces([known_face_encoding], unknown_face_encoding)[0]
        accuracy = face_recognition.face_distance([known_face_encoding], unknown_face_encoding)[0]
        results.append({'Unknown Person': unknown_names[i], 'Known Person': known_names[j], 'Match': match, 'Accuracy': 1 - accuracy})



# %%
# Create dataframe with results
df = pd.DataFrame(results)



# %%
# Convert the 'Known Person' column from string to numeric values
le = LabelEncoder()
df['Known Person'] = le.fit_transform(df['Known Person'])



# %%
# Adjust the confusion matrix according to the numeric values
confusion_matrix = pd.crosstab(df['Known Person'], df['Match'], rownames=['Actual'], colnames=['Predicted'])

# Print the metrics
print('Confusion Matrix:\n', confusion_matrix)
print('Classification Report:\n', classification_report(df['Known Person'], df['Match']))



# %%
# Plot confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='g', ax=ax)
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.show()



# %%
# Plot accuracy by known person
accuracy_by_person = df.groupby('Known Person')['Accuracy'].mean()
sns.barplot(x=accuracy_by_person.index, y=accuracy_by_person.values, color='blue')
plt.title('Acurácia por Pessoa Conhecida')
plt.xlabel('Pessoa Conhecida')
plt.ylabel('Acurácia (%)')
plt.show()



# %%
# Plot distribution of accuracy by known person
sns.boxplot(x='Known Person', y='Accuracy', data=df, color='blue')
plt.title('Distribuição da Acurácia por Pessoa Conhecida')
plt.xlabel('Pessoa Conhecida')
plt.ylabel('Acurácia (%)')
plt.show()



# %%
# Plot histogram
plt.hist(df['Accuracy'], bins=20, color='blue')
plt.title('Distribuição da Acurácia')
plt.xlabel('Acurácia (%)')
plt.ylabel('Frequência')
plt.show()



# %%
fim = datetime.datetime.now()
print('Tempo de execução:', fim - inicio)


# %%

df.to_csv('results.csv', index=False)


