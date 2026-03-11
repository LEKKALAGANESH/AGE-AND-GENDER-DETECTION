 -- Real-Time Age and Gender Recognition System

-- Project Overview

This project implements a Real-Time Age and Gender Recognition System using Deep Learning and Computer Vision.

The system detects human faces from a webcam feed and predicts:

-  Age (Regression)
-  Gender (Binary Classification)

The model is built using MobileNet (Transfer Learning) and TensorFlow, and face detection is performed using OpenCV Haar Cascade.

This project is suitable for:
- Academic submissions
- IEEE documentation
- Resume projects
- Real-time AI demonstrations

---

* Technologies Used

- Python 3.9+
- TensorFlow
- OpenCV
- NumPy
- Scikit-learn
- MobileNet (Pretrained on ImageNet)

---

* Model Architecture

Backbone: MobileNet (Pretrained)

The model has two outputs:

1. Age Output
- Type: Regression
- Activation: Linear
- Loss Function: Mean Absolute Error (MAE)

2. Gender Output
- Type: Binary Classification
- Activation: Sigmoid
- Loss Function: Binary Crossentropy

---

## 📂 Project Structure
AgeGenderRecognition/
│
├── dataset/
│ ├── image1.jpg
│ ├── image2.jpg
│
├── models/
│ └── age_gender_model.h5
│
├── training/
│ └── train_model.py
│
├── inference/
│ └── webcam_app.py
│
├── haarcascade_frontalface_default.xml
├── requirements.txt
└── README.md


---

Dataset Preparation

Recommended Dataset: UTKFace

Image Naming Format: age_gender_race_timestamp.jpg


No subfolders required.

---

## Installation Guide

### Step 1: Clone or Download Project

Download the project folder.

---

### Step 2: Install Dependencies

Open Command Prompt inside the project folder and run: pip install -r requirements.txt


---

## Training the Model

Go inside training folder:cd training
run : python train_model.py


Training Process:
- Loads dataset
- Extracts age and gender from file names
- Splits data into training and testing
- Loads MobileNet pretrained weights
- Trains multi-output model
- Saves model as: models/age_gender_model.h5


---

##  Running Real-Time Webcam Application

Go inside inference folder: cd inference
Run:python webcam_app.py


The webcam will open and display:

- Face bounding box
- Predicted Gender
- Predicted Age

Press Q to exit.

---

##  How the System Works

1. Webcam captures live frames
2. OpenCV detects faces
3. Face is resized to 224x224
4. Image normalized (0–1)
5. MobileNet extracts features
6. Model predicts:
   - Age (numeric)
   - Gender (Male/Female)
7. Results displayed on screen

---

##  Performance Notes

- Optimized for CPU
- Lightweight MobileNet backbone
- Real-time capable
- Age predictions are approximate
- Performance depends on lighting and camera quality

---

##  Troubleshooting

Problem: Webcam not opening  
Solution: Ensure camera permissions are enabled.

Problem: Model file not found  
Solution: Make sure training completed successfully.

Problem: TensorFlow error  
Solution: Use Python 3.9 or 3.10.

---

##  Future Improvements

- Add Emotion Detection
- Improve age accuracy using fine-tuning
- Convert to Web App using Flask
- Add Model Quantization
- Deploy using ONNX
- Add GUI Interface
- Cloud deployment

---

##  Academic Use

This project is intended for:

- Educational purposes
- Academic submission
- Research demonstration

Age predictions are estimations and may not be exact.

---

##  Author

Developed as a Deep Learning and Computer Vision project using Transfer Learning.

---

##  License

This project is for educational and research purposes only.




