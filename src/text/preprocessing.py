import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)       # remove URLs
    text = re.sub(r"@\w+", "", text)          # remove mentions
    text = re.sub(r"#\w+", "", text)          # remove hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)   # remove special chars
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text


def load_data(train_path, val_path, test_path):
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)

    return train, val, test


def preprocess_dataframe(df):
    df["text"] = df["text"].apply(clean_text)
    return df


def encode_labels(train, val, test):
    le = LabelEncoder()

    train["label"] = le.fit_transform(train["label"])
    val["label"] = le.transform(val["label"])
    test["label"] = le.transform(test["label"])

    return train, val, test, le