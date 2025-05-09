
from flask import Flask, request, jsonify
import os
from tensorflow.keras.models import load_model
from joblib import load

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

saved_model = load_model("models/lstm_model.h5")
audio_model = load_model("models/audio_model.h5")
scaler = load("models/scaler.joblib")

@app.route("/")
def home():
    return jsonify({"message": "API is running"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
