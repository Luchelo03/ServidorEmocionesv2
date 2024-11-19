import numpy as np
import tensorflow as tf
import cv2

# Cargar el modelo entrenado
interpreter = tf.lite.Interpreter(model_path="models/model.tflite")
interpreter.allocate_tensors()

# Obtener detalles de entrada y salida del modelo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Lista de etiquetas de emociones
emotions_labels = ["Sorpresa", "Miedo", "Enojado", "Neutral", "Triste", "Disgustado", "Feliz"]

def predict_emotion(frame):
    # Redimensionar y normalizar el frame
    img = cv2.resize(frame, (48, 48))
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
    
    # Pasar el frame al modelo
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    return output[0]  # Vector de probabilidades

def analyze_emotions(frames):
    emotions_per_frame = []
    
    for frame in frames:
        emotion_probabilities = predict_emotion(frame)
        emotions_per_frame.append(emotion_probabilities.tolist())
    
    # Calcular el promedio de emociones en todos los frames
    average_emotions = np.mean(emotions_per_frame, axis=0)
    
    # Crear un diccionario con las emociones promedio
    result = {emotions_labels[i]: float(average_emotions[i]) for i in range(len(average_emotions))}
    
    return emotions_per_frame, result
