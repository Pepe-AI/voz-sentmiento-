import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def preprocessing(audio_path):
    #(----------------------)
    data, sampling_rate = librosa.load(audio_path)

    #calculo del tamaña de la ventana
    duracion_ventana_sec = 20 / 1000  # Convertir ms a segundos
    size_window = duracion_ventana_sec * sampling_rate  # Calcular el tamaño de la ventana en muestras
    

    # Extrae MFCCs
    mfccs = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40, n_fft = size_window) 


    # Calcular la media de los MFCCs para cada archivo
    mfccs_media = np.mean(mfccs, axis=1)


    #Extracción de la Tasa de Cruce por Cero
    zcr = librosa.feature.zero_crossing_rate(data)[0]
    #Sacamos la media
    zcr_mean = np.mean(zcr)


    # Extracción del Pitch
    pitch,_,_ = librosa.pyin(data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    pitch_values = pitch[~np.isnan(pitch)]  # Filtrar NaNs
    median_pitch = get_median(pitch_values)


    # Dividir el nombre del archivo y extraer el tipo de sentimiento
    cut = file.split('-')
    code_sentiment = cut[2]
    intnsity_sentiment = cut[3]
    sentiment_label = emotions.get(code_sentiment, 'desconocido')
    intensity_label = intensity.get(intnsity_sentiment, 'desconocido')





    # Añade a la lista
    data_list["all_pitches"].extend(pitch_values)
    add_to_list(sentiment_label, "all_labels")
    add_to_list(mfccs_media, "all_features")
    add_to_list(zcr_mean, "all_zcr")
    add_to_list(intensity_label, "all_intensity")   
    add_to_list(median_pitch, "all_median_pitch")
    #add_to_list(mel_pca_features, "all_mel_pca")


    return zcr, pitch,data, sampling_rate
    

def bamboo_elbow_method(pca, mfccs):


    pca.fit_transform(mfccs)
    # Calcular la varianza explicada acumulada
    varianza_explicada_acumulada = np.cumsum(pca.explained_variance_ratio_)
    

    # Graficar la varianza explicada acumulada
    plt.figure(figsize=(8, 4))
    plt.plot(varianza_explicada_acumulada, marker='o')
    plt.xlabel('Número de componentes')
    plt.ylabel('Varianza explicada acumulada')
    plt.title('Análisis del Codo para PCA')
    plt.grid(True)
    plt.show()


def analize_pitch(pitch_median_array, median_global_pitch):
    # Definir rangos basados en la mediana y establecer umbrales utilizando el IQR
    Q1 = np.percentile(pitch_median_array, 25)
    Q3 = np.percentile(pitch_median_array, 75)
    IQR = Q3 - Q1


    # Establecer umbrales
    limite_bajo = Q1 - 1.5 * IQR
    limite_alto = Q3 + 1.5 * IQR


    # Reemplaza esto con el valor mínimo real de tus datos de pitch si es mayor que cero
    min_pitch = max(0, np.min(pitch_median_array))


    # Ajusta el límite inferior si es negativo
    limite_bajo_ajustado = max(limite_bajo, min_pitch)
    limite_alto_ajustado = min(limite_alto, max(pitch_median_array)) 

    #obtener limites
    limite_algo_bajo = median_global_pitch - 0.5 * (median_global_pitch - limite_bajo_ajustado)
    limite_algo_alto = median_global_pitch + 0.5 * (limite_alto_ajustado - median_global_pitch) 


    return  Q1, Q3, IQR, limite_bajo, limite_algo_bajo, limite_alto, limite_algo_alto


def create_csv(labels_array,zcr_array,intensity_array,pitch_median_array,frecuency_array,features_array):
    longitud_comun = len(labels_array)
    if not all(len(arr) == longitud_comun for arr in [features_array, zcr_array, intensity_array, pitch_median_array,labels_array,frecuency_array]):
        raise ValueError("Las longitudes de los arrays no coinciden")


    # Crear DataFrames individuales
    df_labels = pd.DataFrame({'Labels': labels_array})
    df_zcr = pd.DataFrame({'ZCR': zcr_array})
    df_intensity = pd.DataFrame({'Intensity': intensity_array})
    df_median_pitch = pd.DataFrame(pitch_median_array, columns=['median_pitch'])
    df_frecuency = pd.DataFrame({'Frecuency': frecuency_array})
    #df_mel_pca = pd.DataFrame({"mel_pca": all_mel_pca})


    columnas_features = [f'Feature_{i+1}' for i in range(features_array.shape[1])]
    df_features = pd.DataFrame(features_array, columns=columnas_features)


    # Concatenar los DataFrames
    #df_mel_pca = pd.DataFrame({"mel_pca": all_mel_pca})
    df_final = pd.concat([df_labels, df_zcr, df_intensity, df_median_pitch, df_frecuency, df_features], axis=1)

    # Guardar el DataFrame como un archivo CSV
    df_final.to_csv('datos_voz.csv', index=False)


def converter_Array_np(list_name):
    return np.array(data_list[list_name])


def add_to_list(element, list_name):
    data_list[list_name].append(element)


def get_median(array):
    median = np.mean(array)
    return median


def get_variance(array):
    variance = np.var(array)
    return variance


def get_range(array):
    range = np.ptp(array)
    return range

# Cargar un archivo
audio_dir = 'Datos\\Audios'


# diccionario de Listas para almacenar todas las características y etiquetas
data_list = {
    "all_features" : [],
    "all_pitches" : [],
    "all_labels" : [],
    "all_zcr" : [],
    "all_intensity" : [],
    "all_median_pitch" : [],
    "frecuency": [],
}


# Diccionario de emociones 
emotions = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


#diccionario de intensidad
intensity = {
    "01": "normal",
    "02": "strong"
}


 # Carga y preprocesar los archivo de audio
for file in os.listdir(audio_dir):
    audio_path = os.path.join(audio_dir, file)
    #(----------------------)
    zcr, pitch,data, sampling_rate = preprocessing(audio_path)

#Normalizar los datos
scaler_mfccs = StandardScaler()
mfccs_normalizados = scaler_mfccs.fit_transform(data_list["all_features"])




# Reducción de dimensionalidad con PCA
pca = PCA(n_components=10)  # Número de componentes para PCA

#calculo del numero de compoenentes optimo
#bamboo_elbow_method(pca,mfccs)
mfccs_pca = pca.fit_transform(mfccs_normalizados)
print("mfccs_pca",type(mfccs_pca))

mfccs_pca, pca.explained_variance_ratio_





# Convierte las listas en arrays de numpy
features_array = mfccs_pca
labels_array = converter_Array_np("all_labels")
zcr_array = converter_Array_np("all_zcr")
intensity_array = converter_Array_np("all_intensity")
pitch_median_array = converter_Array_np("all_median_pitch")
all_pitch_array = converter_Array_np("all_pitches")




#Obtenemos la media global
median_global_pitch = get_median(pitch_median_array)


#(-------------------)
Q1, Q3, IQR, limite_bajo, limite_algo_bajo, limite_alto, limite_algo_alto = analize_pitch(pitch_median_array, median_global_pitch)


# Crear la tabla de referencia
tabla_referencia = {
    'Mediana Global': median_global_pitch,
    'Cuartil Inferior (Q1)': Q1,
    'Cuartil Superior (Q3)': Q3,
    'Rango Intercuartílico (IQR)': IQR,
    'Limite bajo': limite_bajo,
    "Limite algo bajo": limite_algo_bajo,
    'Limite Alto': limite_alto,
    "Limite algo alto": limite_algo_alto,
}


#Asiganacion de la etiqueta del nivel de frecuencia 
for pitch in pitch_median_array:
    if pitch >= tabla_referencia["Limite Alto"]:
        add_to_list("freceuncia alta", "frecuency") 
    if pitch < tabla_referencia["Limite Alto"] and pitch >= tabla_referencia["Limite algo alto"]:
        add_to_list("freceuncia algo alta", "frecuency")   
    if pitch < tabla_referencia["Limite algo alto"] and pitch >= tabla_referencia["Limite algo bajo"]:
        add_to_list("freceuncia normal", "frecuency")   
    if pitch < tabla_referencia["Limite algo bajo"] and pitch >= tabla_referencia["Limite bajo"]:
        add_to_list("Limite Bajo", "frecuency") 

frecuency_array=converter_Array_np("frecuency")




data_single_feature_reshaped = pitch_median_array.reshape(-1, 1)

scaler_median_pitch = StandardScaler()
median_pitch_normalizados = scaler_median_pitch.fit_transform(data_single_feature_reshaped)


create_csv(labels_array,zcr_array,intensity_array,median_pitch_normalizados,frecuency_array,features_array)





