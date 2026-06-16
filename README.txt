README.txt

MULTIMODAL EMOTION RECOGNITION SYSTEM

# PROJECT OVERVIEW

This project is an AI-based **Multimodal Emotion Recognition System** that predicts human emotions using:

* Text Emotion Detection*
* Facial Emotion Detection*
* Fusion-Based Final Decision*

The system combines predictions from both modalities and generates a final emotion using weighted fusion logic.

The project supports the following emotions:

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

The application includes:

* Real-time webcam emotion detection
* Text emotion analysis
* Emotion fusion mechanism
* Web-based frontend using Flask
* Constraint optimization using model quantization

---

# Technologies & Frameworks Used

## Programming Language

* Python

## Frontend

* HTML
* CSS
* JavaScript

## Backend

* Flask

## AI / Deep Learning Libraries

* PyTorch
* Transformers (HuggingFace)
* DeepFace
* OpenCV
* TensorFlow

## NLP Model

* RoBERTa-base
* Fine-tuned on GoEmotions Dataset

## Face Emotion Detection

* DeepFace Emotion Recognition Model

## Optimization Technique

* Dynamic Quantization

---

# Folder Structure

```text
project/
│
├── app/
│   └── app.py
├── data/
│   └── text/
│       └── go_emotions_dataset.csv
│
├── models/
│   └── face_model/
│	└── model.h5
│   └── text_model/
│	└── emotion_roberta/
│       └── quantized_roberta.pth
│	└── label_encoder.pkl
│	└── model.pkl
│	└── vectorizer.pkl
│       
│
├── src/
│   ├── text/
│       └── model.py
│	└── predict.py
│	└── preprocessing.py
│	└── train.py
│   ├── face/
│	└── face_detect.py
│	└── haarcascade_frontalface_default.xml
│	└── model.py
│	└── realtime.py
│	└── train.py
│   ├── fusion/
│	└── fusion_model.py
│	└── predict.py
│   └── optimization/
│	└── measure_constraints.py
│	└── quantize_model.py
│
├── web/
│   ├── templates/
│	└── index.html
│	└── result.html
│   ├── static/
│	└── script.js
│	└── style.css
│   └── app.py
│
│__main.py
│
└──requirements.txt
```

---

# Model Details

## 1. Text Emotion Recognition Model

### Model Used

RoBERTa-base Transformer Model

### Dataset

GoEmotions Dataset

### Number of Classes

7 Emotion Classes

### Working

* Input text is tokenized using RoBERTa tokenizer
* Tokens are passed through transformer encoder layers
* Final classification layer predicts emotion probabilities
* Softmax generates confidence scores

### Output

* Predicted Emotion
* Confidence Score

---

## 2. Face Emotion Recognition Model

### Model Used

DeepFace Emotion Analysis

### Working

* Webcam captures image frame
* Face is detected using Haar Cascade
* DeepFace predicts dominant facial emotion
* Emotion confidence is extracted

### Output

* Facial Emotion
* Confidence Score

---

## 3. Fusion Model

### Fusion Logic

Weighted decision-level fusion

### Weights Used

* Face Model Weight = 0.6
* Text Model Weight = 0.4

### Fusion Rules

1. If both models predict same emotion:

   * Final emotion = common prediction

2. If predictions differ:

   * Higher weighted confidence wins

### Final Output

* Final Emotion
* Final Confidence
* Decision Explanation

---

# Constraint Optimization

## Optimization Technique

Dynamic Quantization

## Objective

Reduce:

* Model Size
* RAM Usage

while maintaining acceptable accuracy.

## Results Measured

* Accuracy
* Model Size
* Inference Time
* Memory Usage

---



# STEPS TO RUN THE CODE

1. Clone the Repository

git clone https://github.com/poojitha-t06/Multimodal-Emotion-Recognition-System.git
cd Multimodal-Emotion-Recognition-System


2. Create Virtual Environment

Windows:
python -m venv emoenv
.\emoenv\Scripts\activate

Linux / Mac:
python3 -m venv emoenv
source emoenv/bin/activate


3. Install Required Packages

pip install -r requirements.txt



4. Download Required Text Models

Download the text model files from:

https://drive.google.com/file/d/1hVJOB6sztSxY0Bbgw96Srrm83RK9UTuC/view?usp=sharing

Extract the downloaded files into:
models/text_model/



5. Run the Project
python web/app.py


6. Open in Browser
http://127.0.0.1:5000




# DEPENDENCIES/ LIBRARIES REQUIRED

Install the following libraries before running the project.

## Main Libraries

```bash
pip install torch
pip install transformers
pip install tensorflow
pip install tf-keras
pip install deepface
pip install opencv-python
pip install flask
pip install numpy
pip install pandas
pip install scikit-learn
pip install psutil
```

---




# EXPECTED OUTPUTS FOR VERIFICATION

## 1. Text Emotion Detection

### Input

```text
I am feeling very happy today!
```

### Expected Output

```text
Happy
```

---

## 2. Face Emotion Detection

### Input

Webcam facial expression

### Expected Output

```text
Detected facial emotion with confidence
```

---

## 3. Fusion Output

### Example

Text Emotion:

Face Emotion:

### Final Output

```text
Final Emotion
Confidence
Decision
```

# Real-Time Features

* Live webcam emotion detection
* Real-time text emotion prediction
* Real-time fusion result
* Confidence score display
* AI decision explanation

