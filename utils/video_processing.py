import cv2

# Cargar los clasificadores Haar
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")

def process_video_frames(video_path, frame_limit=100):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, frame_count // frame_limit)  # Tomar hasta frame_limit frames
    frames = []
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = detect_face(frame_rgb)
            if face is not None:
                frames.append(face)  # Solo añadir el rostro detectado
        frame_idx += 1
    
    cap.release()
    return frames

def detect_face(frame):
    # Convertir el frame a escala de grises para la detección
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Si se detecta al menos un rostro, extraer el primer rostro detectado
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]  # Recortar la región del rostro
        return cv2.resize(face, (48, 48))  # Redimensionar el rostro a 48x48
    return None  # Si no se detecta rostro, devolver None
