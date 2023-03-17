import face_recognition as fr, os, pandas as pd, time

def reconhece_face(url_foto):
    foto = fr.load_image_file(url_foto)
    rostos = fr.face_encodings(foto)
    if(len(rostos) > 0):
        return True, rostos
    return False, []

def monta_arquivo():
    armazenado = r'C:\Users\bruno\Downloads\FaceRecognition_v2\FaceRecognition_v2\src\assets\conhecidos'
    rostos_conhecidos = []
    nomes_dos_rostos = []
    
    for pessoa in os.listdir(armazenado):
        foto_pessoa = f'{armazenado}/{pessoa}'
        nome_pessoa = pessoa.split('.')[0]
        foto = reconhece_face(foto_pessoa)
        if foto[0]:
            rostos_conhecidos.append(foto_pessoa)
            nomes_dos_rostos.append(nome_pessoa)
        df = pd.DataFrame({'Foto': rostos_conhecidos, 'Nome': nomes_dos_rostos})
        df.to_csv(r'C:\Users\bruno\Downloads\FaceRecognition_v2\FaceRecognition_v2\src\assets\rostos_conhecidos.csv', index=False)

def get_rostos():
    rostos_conhecidos = []
    nomes_dos_rostos = []
    df = pd.read_csv(r'C:\Users\bruno\Downloads\FaceRecognition_v2\FaceRecognition_v2\src\assets\rostos_conhecidos.csv')
    for indice, linha in df.iterrows():
        foto = reconhece_face(fr"{linha['Foto']}")
        if(foto[0]):
            rostos_conhecidos.append(foto[1][0])
            nomes_dos_rostos.append(fr"{linha['Nome']}")
    
    return rostos_conhecidos, nomes_dos_rostos   