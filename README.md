# Facial Recognition Attendance System

Welcome to the **Facial Recognition Attendance System**, a Flask-based web application that automates attendance marking using facial recognition technology. This system leverages deep learning models (FaceNet and MTCNN) to identify students and record their attendance for specific class periods, storing the data in a PostgreSQL database. The system allows users to capture images, align datasets, populate the database, manage courses, and view attendance records dynamically via AJAX.

## Features

- **Facial Recognition**: Uses MTCNN for face detection and FaceNet with a custom classifier for face recognition.
- **Attendance Marking**: Automatically marks students as present or absent during a 5-minute window, with unmarked students marked as absent afterward.
- **Web Interface**: Built with Flask, featuring a user-friendly interface to trigger attendance and view records.
- **Dynamic Updates**: Implements AJAX to fetch and display attendance records without page reloads.
- **Database Integration**: Stores student, course, and attendance data in a PostgreSQL database.
- **Data Management**: Supports capturing raw images, aligning datasets, and populating the database with student information.

## Prerequisites

- **Python 3.10**
- **PostgreSQL** (with a database named `facial_recognition`)
- **OpenCV** (`cv2`)
- **PyTorch**
- **FaceNet-PyTorch** (`facenet-pytorch`)
- **Pillow** (`PIL`)
- **psycopg2**
- **Flask**
- **NumPy**
- **pandas**
- **imageio**

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Gyanhari/Seventh-Sem-Project.git
cd Seventh-Sem-Project
```

### 2. Set Up a Virtual Environment

```bash
python -m venv venv
Windows: venv\Scripts\activate
```

### 3. Install Dependencies

Install the required Python packages:

```
pip install -r facial_recognition/requirements.txt
```

### 4. Configure PostgreSQL

Install PostgreSQL and create a database named facial_recognition.

```sql
psql -U postgres -d facial_recognition < mydatabase_backup.sql
```

Update the DB_CONFIG dictionary in app.py with your PostgreSQL credentials:

```python

DB_CONFIG = {
'host': 'localhost',
'user': 'postgres',
'password': 'your_password',
'database': 'facial_recognition',
'port': '5432'
}
```

### 5. Prepare the Dataset

Create the following directory structure:

```bash
facial_recognition/
├── dataset/
│ ├── raw/ # Raw captured images
│ ├── aligned/ # Aligned images for recognition
│ └── failed/ # Failed alignment attempts
├── embeddings/ # Saved embeddings and label encoder
├── models/ # Saved classifier model
```

Capture raw images using the /capture route or place pre-existing images in the raw folder.

Align the dataset using the /align route to generate aligned faces in the aligned folder.

Train the classifier and save the model weights (best_classifier.pth) and label encoder (label_encoder.pkl) in the models and embeddings directories, respectively.

### 6. Run the Application

```bash

python app.py

```

Open your browser and navigate to http://127.0.0.1:5000/.

## Usage

### 1. Capture Images

Visit /capture to capture 100 frames of a student's face using the webcam.

Enter the student's name and a 6-digit roll number, then submit.

### 2. Align Dataset

Visit /align to align the raw images into a standardized format for recognition.

Aligned images are moved to the aligned folder, and failed attempts go to failed.

### 3. Populate Database

Visit /populate to add student data from the aligned dataset to the Students table in the database.

### 4. Manage Courses

Visit /course to add new courses or class periods.

Select an existing course or create a new one, then specify the date, start time, and duration (45-120 minutes).

### 5. Train the Model

Visit /train to train the model and see it's accuracy and loss graph.

### 6. Take Attendance

Visit /attendance to select a period and click "Trigger Attendance".

The webcam runs for 5 minutes, marking recognized students as present and others as absent.

Click "View Attendance" to see records dynamically updated via AJAX.

### Project Structure

```bash
facial_recognition/
├── app.py # Main Flask application
├── facial_recognition/ # Subdirectory for dataset and models
│ ├── dataset/
│ │ ├── raw/ # Raw captured images
│ │ ├── aligned/ # Aligned images for recognition
│ │ └── failed/ # Failed alignment attempts
│ ├── embeddings/ # Saved embeddings and label encoder
│ ├── models/ # Saved classifier model
│ └── src/ # Helper scripts (e.g., align_dataset.py, populate_database.py)
├── templates/ # HTML templates
│ ├── index.html
│ ├── capture.html
│ ├── align.html
│ ├── train.html
│ ├── populate.html
│ ├── course.html
│ └── attendance.html
└── README.md # This file
```

### Configuration

#### RAW_DATA_PATH: Directory for raw images (facial_recognition/dataset/raw).

#### ALIGNED_DATA_PATH: Directory for aligned images (facial_recognition/dataset/aligned).

#### FAILED_DATA_PATH: Directory for failed alignments (facial_recognition/dataset/failed).

#### UPLOADED_USERS_FILE: Text file to track uploaded users (uploaded_users.txt).

#### Device: Uses CUDA if available, otherwise CPU.
