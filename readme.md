# 🧠 Multimodal Emotion Recognition System

A complete AI-based system that detects human emotions using **Text + Facial Expressions** in real-time.
This project combines **Natural Language Processing (NLP)** and **Computer Vision (CV)** to improve emotion detection accuracy through a multimodal approach.

---

## 🚀 Features

* 📝 Text Emotion Detection (TF-IDF + Logistic Regression)
* 📷 Real-time Face Emotion Detection (CNN + OpenCV)
* 🔗 Multimodal Fusion (Text + Face)
* 🌐 Web Application (Flask + JavaScript)
* 🎥 Live Webcam inside Browser
* 📊 Clean UI with Emotion Results

---
# How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/poojitha-t06/Multimodal-Emotion-Recognition-System.git

cd Multimodal-Emotion-Recognition-System
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv emoenv

.\emoenv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv emoenv

source emoenv/bin/activate
```

---

## 3. Install Required Packages

```bash
pip install -r requirements.txt
```

---

## 4. Download Required Text Models

Download the text model files from:

https://drive.google.com/file/d/1hVJOB6sztSxY0Bbgw96Srrm83RK9UTuC/view?usp=sharing

Extract the downloaded files into:

```bash
models/text_model/
```

---

## 5. Run the Project

```bash
python web/app.py
```

---

## 6. Open in Browser

```bash
http://127.0.0.1:5000
```