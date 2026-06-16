import sys
import os
import base64
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify

# Fix imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.text.predict import predict_emotion
from src.face.realtime import predict_emotion as face_predict
from src.fusion.fusion_model import fuse_emotions

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        text = data.get("text")
        image_data = data.get("image")

        # TEXT EMOTION
        
        text_result = predict_emotion(text) if text else ("Neutral", 0.0)

        # Default face result
        face_result = ("Neutral", 0.0)

        if image_data:
            image_data = image_data.split(",")[1]
            img_bytes = base64.b64decode(image_data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Convert to gray
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect face
            face_cascade = cv2.CascadeClassifier(
                "src/face/haarcascade_frontalface_default.xml"
            )

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face_result = face_predict(face)
                break  # take first face

        # FUSION
        final_result = fuse_emotions(face_result, text_result)

        return jsonify({

            "text_emotion": text_result[0],
            "text_confidence": float(round(text_result[1], 3)),

            "face_emotion": face_result[0],
            "face_confidence": float(round(face_result[1], 3)),

            "final_emotion": final_result["final_emotion"],
            "final_confidence": float(final_result["confidence"]),

            "decision": final_result["decision"]
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)