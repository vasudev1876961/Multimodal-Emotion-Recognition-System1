import re
import numpy as np


# ===============================
# TEXT HELPERS
# ===============================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ===============================
# IMAGE HELPERS
# ===============================

def preprocess_face(face_img):
    import cv2

    face_img = cv2.resize(face_img, (48, 48))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = face_img / 255.0
    face_img = np.reshape(face_img, (1, 48, 48, 1))

    return face_img


# ===============================
# FUSION HELPERS
# ===============================

def normalize_emotion_label(label):
    """
    Ensures consistent format across models
    (lowercase for fusion comparison)
    """
    return label.lower()


def print_result(text_emotion, face_emotion, final_emotion):
    print("\n==============================")
    print(f"Text Emotion  : {text_emotion}")
    print(f"Face Emotion  : {face_emotion}")
    print(f"Final Emotion : {final_emotion}")
    print("==============================\n")