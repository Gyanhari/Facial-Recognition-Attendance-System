import cv2
import os
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import torch.nn as nn
import pandas as pd
from datetime import datetime
import logging
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load FaceNet model
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load classifier
class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# Load class names and initialize classifier
embeddings_dir = 'facial_recognition/embeddings'
with open(os.path.join(embeddings_dir, 'label_encoder.pkl'), 'rb') as f:
    le = pickle.load(f)
class_names = le.classes_
num_classes = len(class_names)
classifier = Classifier(num_classes).to(device)
classifier.load_state_dict(torch.load('facial_recognition/models/best_classifier.pth', map_location=device))
classifier.eval()

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device=device)

attendance_file = 'facial_recognition/attendance.csv'

def load_or_create_attendance_file():
    """Load the attendance file or create it if it doesn't exist."""
    if not os.path.exists(attendance_file):
        logging.info(f"Creating new attendance file: {attendance_file}")
        df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
        df.to_csv(attendance_file, index=False)
    else:
        logging.info(f"Loading existing attendance file: {attendance_file}")
        df = pd.read_csv(attendance_file)
    return df

def mark_attendance(name):
    """Mark attendance for a recognized face."""
    try:
        # Validate name
        if not name or not isinstance(name, str):
            logging.error("Invalid name provided.")
            return

        # Get current date and time
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H:%M:%S')

        # Load or create attendance file
        df = load_or_create_attendance_file()

        # Debug: Print DataFrame before update
        logging.debug(f"DataFrame before update:\n{df}")

        # Check if the student has already been marked present today
        if not ((df['Name'] == name) & (df['Date'] == current_date)).any():
            # Add a new attendance record
            new_entry = pd.DataFrame({'Name': [name], 'Date': [current_date], 'Time': [current_time]})
            df = pd.concat([df, new_entry], ignore_index=True)

            # Debug: Print DataFrame after update
            logging.debug(f"DataFrame after update:\n{df}")

            # Save to CSV
            df.to_csv(attendance_file, index=False)
            logging.info(f"Attendance marked for {name} on {current_date} at {current_time}")
        else:
            logging.info(f"{name} has already been marked present for today.")

    except Exception as e:
        logging.error(f"Error marking attendance for {name}: {e}")
# Function to recognize a face
def recognize_face(face_tensor):
    """Recognize a face using the trained classifier."""
    with torch.no_grad():
        embedding = facenet(face_tensor.unsqueeze(0).to(device))
        outputs = classifier(embedding)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        best_class_idx = torch.argmax(probabilities).item()
        best_class_prob = probabilities[0, best_class_idx].item()
        return class_names[best_class_idx], best_class_prob

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logging.error("Error: Could not open webcam.")
    exit()

logging.info("Press 'q' to quit the application.")

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            logging.error("Error: Failed to capture frame.")
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(frame_rgb)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(coord) for coord in box]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                face = frame_rgb[y1:y2, x1:x2]

                if face.shape[0] > 0 and face.shape[1] > 0:
                    face_pil = Image.fromarray(face)

                    # Detect and preprocess the face with MTCNN
                    face_cropped = mtcnn(face_pil)

                    # Check if MTCNN successfully crops a face
                    if face_cropped is not None and len(face_cropped) > 0:
                        try:
                            name, prob = recognize_face(face_cropped[0])
                            label = f"{name} ({prob:.2f})" if prob > 0.5 else "Unknown"
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                            if prob > 0.7:
                                mark_attendance(name)
                            else:
                                logging.info("No face detected or confidence too low.")
                        except Exception as e:
                            logging.error(f"Error recognizing face: {e}")

                # Draw bounding box on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow('Webcam Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Exiting application.")
            break

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        continue

cap.release()
cv2.destroyAllWindows()