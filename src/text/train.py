import os
import joblib

from src.text.preprocessing import load_data, preprocess_dataframe, encode_labels
from src.text.model import create_vectorizer, create_model

def train():
    # Paths
    train_path = "data/text/train.csv"
    val_path = "data/text/val.csv"
    test_path = "data/text/test.csv"

    # Load data
    train_df, val_df, test_df = load_data(train_path, val_path, test_path)

    # Preprocess
    train_df = preprocess_dataframe(train_df)
    val_df = preprocess_dataframe(val_df)
    test_df = preprocess_dataframe(test_df)

    # Encode labels
    train_df, val_df, test_df, label_encoder = encode_labels(train_df, val_df, test_df)

    # Vectorization
    vectorizer = create_vectorizer()

    X_train = vectorizer.fit_transform(train_df["text"])
    X_val = vectorizer.transform(val_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    y_train = train_df["label"]
    y_val = val_df["label"]
    y_test = test_df["label"]

    # Model
    model = create_model()
    model.fit(X_train, y_train)

    # Create model directory
    os.makedirs("models/text_model", exist_ok=True)

    # Save model
    joblib.dump(model, "models/text_model/model.pkl")
    joblib.dump(vectorizer, "models/text_model/vectorizer.pkl")
    joblib.dump(label_encoder, "models/text_model/label_encoder.pkl")

    print("✅ Model trained and saved successfully!")


if __name__ == "__main__":
    train()