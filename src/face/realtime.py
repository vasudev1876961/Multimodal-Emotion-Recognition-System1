import cv2
from deepface import DeepFace

# Emotion labels mapping
emotion_map = {
    "angry": "Angry",
    "disgust": "Disgust",
    "fear": "Fear",
    "happy": "Happy",
    "sad": "Sad",
    "surprise": "Surprise",
    "neutral": "Neutral"
}


# -------------------------------
# Predict emotion from face image
# -------------------------------


# -------------------------------
# Realtime webcam testing
# -------------------------------
def run_realtime():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        emotion, confidence = predict_emotion(frame)

        cv2.putText(
            frame,
            f"Emotion: {emotion} ({confidence:.2f})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("DeepFace Emotion Detection", frame)

        # Exit on q OR window close
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.getWindowProperty("DeepFace Emotion Detection", cv2.WND_PROP_VISIBLE) < 1:
            break


    cap.release()
    cv2.destroyAllWindows()


# -------------------------------
# Used for fusion
# -------------------------------
def predict_emotion(face_img):

    result = DeepFace.analyze(
        face_img,
        actions=['emotion'],
        enforce_detection=False
    )

    emotions = result[0]['emotion']

    emotion = max(emotions, key=emotions.get)

    confidence = emotions[emotion] / 100.0

    return emotion.capitalize(), confidence


if __name__ == "__main__":
    run_realtime()