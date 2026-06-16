import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer + model
MODEL_PATH = "models/text_model/emotion_roberta"

tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)

model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)

model.to(device)
model.eval()

# Label mapping
id_to_label = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}


def predict_emotion(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)

    confidence, pred = torch.max(probs, dim=1)

    emotion = id_to_label[pred.item()]

    return emotion, confidence.item()



if __name__ == "__main__":
    while True:
        text = input("Enter text: ")

        if text.lower() == "exit":
            break

        emotion = predict_emotion(text)

        print("Predicted Emotion:", emotion)