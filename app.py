from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from utils.video_processing import process_video_frames
from utils.emotion_analysis import analyze_emotions, emotions_labels  # Importar emotions_labels

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

@app.route("/")
def home():
    return "¡Servidor funcionando correctamente en Render!"


@app.route("/upload_video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video = request.files["video"]
    filename = secure_filename(video.filename)
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    video.save(video_path)

    # Procesar el video y obtener las emociones promedio
    frames = process_video_frames(video_path)
    if not frames:
        return jsonify({"error": "No faces detected in the video"}), 400  # Manejo de error si no se detectan rostros

    emotions_per_frame, average_emotions = analyze_emotions(frames)

    # Guardar el registro de probabilidades de cada frame en un archivo de texto
    log_path = os.path.join(app.config["UPLOAD_FOLDER"], "emotion_analysis_log.txt")
    with open(log_path, "w") as f:
        for i, probs in enumerate(emotions_per_frame):
            probs_str = ", ".join([f"{emotions_labels[j]}: {probs[j]:.4f}" for j in range(len(probs))])
            f.write(f"Frame {i + 1}: {probs_str}\n")

    os.remove(video_path)  # Eliminar el video después de procesarlo
    return jsonify(average_emotions)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Usar el puerto 5000 como predeterminado
    app.run(host="0.0.0.0", port=port, debug=True)
