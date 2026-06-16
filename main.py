from src.text.predict import predict_emotion as text_predict
from src.face.realtime import get_face_emotion
from src.fusion.fusion_model import fuse_emotions

from utils.helper import print_result


def main():
    print("===== MULTIMODAL EMOTION AI =====")

    while True:
        text_input = input("\nEnter text (or 'exit'): ")

        if text_input.lower() == "exit":
            print("Exiting...")
            break

        # 🔹 TEXT EMOTION
        text_emotion = text_predict(text_input)

        # 🔹 FACE EMOTION (REALTIME)
        print("Opening webcam... Press 'c' to capture emotion")
        face_emotion = get_face_emotion()

        if face_emotion is None:
            print("No face detected. Try again.")
            continue

        # 🔹 FUSION
        final_emotion = fuse_emotions(face_emotion, text_emotion)

        # 🔹 DISPLAY RESULT
        print_result(text_emotion, face_emotion, final_emotion)


if __name__ == "__main__":
    main()