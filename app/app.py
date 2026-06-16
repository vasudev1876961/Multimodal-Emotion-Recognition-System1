import streamlit as st
import sys
import os
import numpy as np

# Fix imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.text.predict import load_models as load_text_models
from src.face.realtime import get_face_emotion, predict_emotion
from src.fusion.fusion_model import fuse_emotions
from utils.helper import clean_text

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Emotion AI", layout="wide")

# ===============================
# TITLE
# ===============================
st.title("🧠 Multimodal Emotion Recognition System")
st.markdown("### Text + Face Emotion Detection")

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Choose Mode", ["Text Emotion", "Face Emotion", "Fusion (Text + Face)"])

# ===============================
# LOAD TEXT MODEL ONCE
# ===============================
text_model, vectorizer, label_encoder = load_text_models()


# ===============================
# TEXT EMOTION SECTION
# ===============================
if mode == "Text Emotion":

    st.header("📝 Text Emotion Detection")

    user_text = st.text_area("Enter text here:")

    if st.button("Analyze Text"):
        if user_text.strip() == "":
            st.warning("Please enter text")
        else:
            text = clean_text(user_text)
            vec = vectorizer.transform([text])

            probs = text_model.predict_proba(vec)[0]
            pred = np.argmax(probs)

            emotion = label_encoder.inverse_transform([pred])[0]

            st.success(f"Predicted Emotion: {emotion}")

            # Show confidence
            st.subheader("Confidence Scores")
            st.bar_chart(probs)


# ===============================
# FACE EMOTION SECTION
# ===============================
elif mode == "Face Emotion":

    st.header("📷 Face Emotion Detection")

    if st.button("Open Camera"):
        st.warning("Press 'c' to capture emotion from webcam")

        emotion = get_face_emotion()

        if emotion:
            st.success(f"Detected Emotion: {emotion}")
        else:
            st.error("No face detected")


# ===============================
# FUSION SECTION
# ===============================
elif mode == "Fusion (Text + Face)":

    st.header("🔗 Multimodal Fusion")

    user_text = st.text_area("Enter text:")

    if st.button("Analyze Both"):

        if user_text.strip() == "":
            st.warning("Enter text first")
        else:
            # TEXT
            text = clean_text(user_text)
            vec = vectorizer.transform([text])
            probs = text_model.predict_proba(vec)[0]
            pred = np.argmax(probs)
            text_emotion = label_encoder.inverse_transform([pred])[0]

            st.info(f"Text Emotion: {text_emotion}")

            # FACE
            st.warning("Opening webcam... Press 'c'")
            face_emotion = get_face_emotion()

            if not face_emotion:
                st.error("Face not detected")
            else:
                st.info(f"Face Emotion: {face_emotion}")

                # FUSION
                final_emotion = fuse_emotions(face_emotion, text_emotion)

                st.subheader("🎯 Final Emotion")
                st.success(final_emotion)

                # Confidence chart
                st.subheader("Text Confidence")
                st.bar_chart(probs)