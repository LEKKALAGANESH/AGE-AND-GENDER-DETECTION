import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load Models
age_model = load_model("models/age_model.h5")
gender_model = load_model("models/gender_model.h5")

# Gender labels
gender_labels = ['Male', 'Female']

# Start Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not re
    t:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = face / 255.0
        face = np.reshape(face, (1, 224, 224, 3))

        # Predict Age
        age_pred = age_model.predict(face)
        age = int(age_pred[0])

        # Predict Gender
        gender_pred = gender_model.predict(face)
        gender = gender_labels[int(gender_pred[0] > 0.5)]

        # Draw Rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display Text
        text = f"{gender}, {age} yrs"
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)

    cv2.imshow("Age & Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()