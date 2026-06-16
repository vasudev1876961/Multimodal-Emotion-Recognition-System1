from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def create_vectorizer():
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )
    return vectorizer


def create_model():
    model = LogisticRegression(max_iter=200)
    return model