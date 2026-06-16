from src.text.predict import predict_emotion as text_predict
from src.fusion.fusion_model import fuse_emotions
from src.face.realtime import get_face_emotion


def main():
    while True:
        text_input = input("Enter text (or 'exit'): ")

        if text_input.lower() == "exit":
            break

        # Text prediction
        text_emotion = text_predict(text_input)

        # Face prediction (REALTIME)
        print("Opening webcam... Press 'c' to capture emotion")
        face_emotion = get_face_emotion()

        # Fusion
        final_emotion = fuse_emotions(face_emotion, text_emotion)

        print("\n--- RESULT ---")
        print(f"Text Emotion : {text_emotion}")
        print(f"Face Emotion : {face_emotion}")
        print(f"Final Emotion: {final_emotion}")
        print("--------------\n")


if __name__ == "__main__":
    main()