import os
import time
import psutil
import torch
import pandas as pd
import numpy as np
import sys
import os
import torch
from transformers import RobertaForSequenceClassification
from transformers import RobertaTokenizer

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)
from src.text.predict import predict_emotion

from sklearn.metrics import accuracy_score

from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification
)

# --------------------------------
# LOAD MODEL
# --------------------------------
MODEL_PATH = "models/text_model/emotion_roberta"

tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)

model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

# --------------------------------
# LABELS
# --------------------------------
id_to_label = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# --------------------------------
# MODEL SIZE
# --------------------------------
def get_model_size():

    total_size = 0

    for dirpath, dirnames, filenames in os.walk(MODEL_PATH):

        for f in filenames:

            fp = os.path.join(dirpath, f)

            total_size += os.path.getsize(fp)

    return total_size / (1024 * 1024)


# --------------------------------
# INFERENCE TIME
# --------------------------------
def measure_inference():

    text = "I am excited."

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    start = time.time()

    with torch.no_grad():
        outputs = model(**inputs)

    end = time.time()

    return (end - start) * 1000


# --------------------------------
# MEMORY USAGE
# --------------------------------
def memory_usage():

    process = psutil.Process(os.getpid())

    mem = process.memory_info().rss / (1024 * 1024)

    return mem


# --------------------------------
# ACCURACY
# --------------------------------
def evaluate_accuracy(use_quantized=False):

    df = pd.read_csv("data/text/go_emotions_dataset.csv")

    emotion_cols = [
        'anger',
        'disgust',
        'fear',
        'joy',
        'sadness',
        'surprise',
        'neutral'
    ]

    id_to_label = {
        0: "Angry",
        1: "Disgust",
        2: "Fear",
        3: "Happy",
        4: "Sad",
        5: "Surprise",
        6: "Neutral"
    }

    # Load model
    if use_quantized:
        model = load_quantized_model()
    else:
        model = RobertaForSequenceClassification.from_pretrained(
            MODEL_PATH
        )

    model.eval()

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)

    correct = 0
    total = 0

    sample_df = df.sample(100, random_state=42)

    for _, row in sample_df.iterrows():

        text = row["text"]

        values = row[emotion_cols].values

        if sum(values) == 0:
            continue

        label_id = np.argmax(values)

        true_label = id_to_label[label_id]

        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)

        pred = torch.argmax(outputs.logits, dim=1).item()

        predicted_label = id_to_label[pred]

        if predicted_label == true_label:
            correct += 1

        total += 1

    accuracy = (correct / total) * 100

    return accuracy

def load_quantized_model():

    model = RobertaForSequenceClassification.from_pretrained(
        "models/text_model/emotion_roberta"
    )

    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    quantized_model.load_state_dict(
        torch.load(
            "models/text_model/quantized_roberta.pth",
            map_location=torch.device("cpu")
        )
    )

    quantized_model.eval()

    return quantized_model

def measure_after_optimization():

    process = psutil.Process()

    start_ram = process.memory_info().rss / (1024 * 1024)

    start = time.time()

    model = load_quantized_model()

    sample_text = "I am very happy today"

    tokenizer = RobertaTokenizer.from_pretrained(
        "models/text_model/emotion_roberta"
    )

    inputs = tokenizer(
        sample_text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        _ = model(**inputs)

    end = time.time()

    end_ram = process.memory_info().rss / (1024 * 1024)

    inference_time = (end - start) * 1000

    ram_usage = end_ram - start_ram

    model_size = os.path.getsize(
        "models/text_model/quantized_roberta.pth"
    ) / (1024 * 1024)

    return {
        "model_size": model_size,
        "inference_time": inference_time,
        "ram_usage": ram_usage
    }

# --------------------------------
# MAIN
# --------------------------------
print("\n----- BEFORE OPTIMIZATION -----\n")

before_accuracy = evaluate_accuracy(use_quantized=False)

print(f"Accuracy: {before_accuracy:.2f}%")
print(f"Model Size: {get_model_size():.2f} MB")
print(f"Inference Time: {measure_inference():.2f} ms")
print(f"RAM Usage: {memory_usage():.2f} MB")


print("\n----- AFTER OPTIMIZATION -----\n")

after = measure_after_optimization()

after_accuracy = evaluate_accuracy(use_quantized=True)

print(f"Accuracy: {after_accuracy:.2f}%")
print(f"Model Size: {after['model_size']:.2f} MB")
print(f"Inference Time: {after['inference_time']:.2f} ms")
print(f"RAM Usage: {after['ram_usage']:.2f} MB")