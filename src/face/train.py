import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from src.face.model import create_model


def train():
    train_dir = "data/face/train"
    test_dir = "data/face/test"

    # Image preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=64,
        class_mode='categorical'
    )

    test_data = test_datagen.flow_from_directory(
        test_dir,
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=64,
        class_mode='categorical'
    )

    # Model
    model = create_model()

    # Save best model
    os.makedirs("models/face_model", exist_ok=True)

    checkpoint = ModelCheckpoint(
        "models/face_model/model.h5",
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    # Train
    model.fit(
        train_data,
        validation_data=test_data,
        epochs=15,
        callbacks=[checkpoint]
    )

    print("✅ Face model trained and saved!")


if __name__ == "__main__":
    train()