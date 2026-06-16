# ===============================
# PATH CONFIGURATION
# ===============================

# TEXT DATA
TRAIN_DATA_PATH = "data/text/train.csv"
VAL_DATA_PATH = "data/text/val.csv"
TEST_DATA_PATH = "data/text/test.csv"

# FACE DATA
FACE_TRAIN_DIR = "data/face/train"
FACE_TEST_DIR = "data/face/test"

# MODELS
TEXT_MODEL_PATH = "models/text_model/model.pkl"
TEXT_VECTORIZER_PATH = "models/text_model/vectorizer.pkl"
TEXT_LABEL_ENCODER_PATH = "models/text_model/label_encoder.pkl"

FACE_MODEL_PATH = "models/face_model/model.h5"

# HAARCASCADE
HAARCASCADE_PATH = "src/face/haarcascade_frontalface_default.xml"


# ===============================
# MODEL CONFIG
# ===============================

IMAGE_SIZE = (48, 48)
NUM_CLASSES = 7

# EMOTION LABELS (FACE)
EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]