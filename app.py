import os
import cv2
import time
import random
import numpy as np
import torch
from PIL import Image
import imageio
from facenet_pytorch import MTCNN, InceptionResnetV1
from concurrent.futures import ThreadPoolExecutor, as_completed
from facial_recognition.src.helper import get_dataset, check_rollno_in_aligned
from facial_recognition.src.align_dataset import align_images
import psycopg2
from psycopg2 import Error
from datetime import datetime
import time
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from facial_recognition.src.populate_databse import populate_students
import torch.nn as nn
import pandas as pd
import logging
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': '770312',
    'database': 'facial_recognition',
    'port': '5432'
}

# Paths and configurations
RAW_DATA_PATH = os.path.join('facial_recognition', 'dataset', 'raw')
ALIGNED_DATA_PATH = os.path.join('facial_recognition', 'dataset', 'aligned')
FAILED_DATA_PATH = os.path.join('facial_recognition', 'dataset', 'failed')
UPLOADED_USERS_FILE = 'uploaded_users.txt'

# FaceNet and MTCNN setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device, min_face_size=20, thresholds=[0.6, 0.7, 0.7])

# Load FaceNet model
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Define Classifier
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

# Load class names and classifier
embeddings_dir = 'facial_recognition/embeddings'
try:
    with open(os.path.join(embeddings_dir, 'label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)
    class_names = le.classes_
    num_classes = len(class_names)
    classifier = Classifier(num_classes).to(device)
    classifier.load_state_dict(torch.load('facial_recognition/models/best_classifier.pth', map_location=device))
    classifier.eval()
except Exception as e:
    logging.error(f"Error loading classifier or label encoder: {e}")
    class_names = []
    classifier = None

config = {
    "input_dir": "facial_recognition/dataset/raw",
    "output_dir": "facial_recognition/dataset/aligned",
    "failed_dir": "facial_recognition/dataset/failed",
    "image_size": 182,
    "crop_size": 160,
    "margin": 44,
    "random_order": True,
    "detect_multiple_faces": False,
    "gpu_memory_fraction": 1.0,
}

# Database connection
def get_db_connection():
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Error as e:
        print(f"Error connecting to database: {e}")
        return None

def load_uploaded_users():
    uploaded_users = set()
    if os.path.exists(UPLOADED_USERS_FILE):
        with open(UPLOADED_USERS_FILE, 'r') as file:
            uploaded_users = set(line.strip() for line in file if line.strip())
    return uploaded_users

def save_uploaded_user(rollno):
    with open(UPLOADED_USERS_FILE, 'a') as file:
        file.write(f"{rollno}\n")

def get_periods():
    connection = get_db_connection()
    periods = []
    if connection:
        cursor = connection.cursor()
        cursor.execute("""
            SELECT cp.period_id, c.course_name, cp.period_date, cp.start_time
            FROM Class_Periods cp
            JOIN Courses c ON cp.course_id = c.course_id
            ORDER BY cp.period_date, cp.start_time
        """)
        periods = cursor.fetchall()
        cursor.close()
        connection.close()
    return periods

def recognize_face(face_tensor):
    if classifier is None:
        logging.error("Classifier not loaded.")
        return None, 0.0
    with torch.no_grad():
        embedding = facenet(face_tensor.unsqueeze(0).to(device))
        outputs = classifier(embedding)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        best_class_idx = torch.argmax(probabilities).item()
        best_class_prob = probabilities[0, best_class_idx].item()
        return class_names[best_class_idx], best_class_prob

def mark_attendance(student_id, period_id, connection, is_within_time_limit):
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT status FROM Attendance WHERE student_id = %s AND period_id = %s", (student_id, period_id))
        existing_status = cursor.fetchone()

        if not existing_status:
            status = 'present' if is_within_time_limit else 'absent'
            cursor.execute(
                """
                INSERT INTO Attendance (student_id, period_id, status, recorded_timestamp)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (student_id, period_id) DO NOTHING
                """,
                (student_id, period_id, status)
            )
            connection.commit()
            logging.info(f"Successfully marked student ID {student_id} as {status} for period {period_id}")
            flash(f"Marked student ID {student_id} as {status}", "success")
        else:
            logging.info(f"Student ID {student_id} already marked with status {existing_status[0]} for period {period_id}")

    except Exception as e:
        logging.error(f"Error marking attendance for student ID {student_id}: {e}")
        connection.rollback()
    finally:
        cursor.close()


@app.route('/mark_attendance/<int:period_id>', methods=['POST'])  # Change to POST for AJAX
def mark_attendance_route(period_id):
    if classifier is None or (class_names is None or len(class_names) == 0):
        return jsonify({"status": "error", "message": "Facial recognition models not loaded or class names empty."}), 500

    connection = get_db_connection()
    if not connection:
        return jsonify({"status": "error", "message": "Error connecting to database."}), 500

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        connection.close()
        return jsonify({"status": "error", "message": "Error: Could not open camera."}), 500

    time_limit = 300
    recognized_students = set()
    start_time = time.time()

    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > time_limit:
            logging.info("5-minute time limit reached. Stopping webcam.")
            break

        ret, frame = cap.read()
        if not ret:
            logging.error("Error: Failed to capture frame.")
            break

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
                    face_cropped = mtcnn(face_pil)

                    if face_cropped is not None and len(face_cropped) > 0:
                        try:
                            name, prob = recognize_face(face_cropped[0])
                            if name is None:
                                continue
                            name_for_query = name.replace('_', ' ')
                            label = f"{name} ({prob:.2f})" if prob > 0.7 else "Unknown"
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                            if prob > 0.7:
                                cursor = connection.cursor()
                                cursor.execute(
                                    """
                                    SELECT student_id 
                                    FROM Students 
                                    WHERE CONCAT(first_name, ' ', COALESCE(middle_name, ''), ' ', last_name) ILIKE %s
                                    """,
                                    (name_for_query,)
                                )
                                student = cursor.fetchone()
                                if student:
                                    student_id = student[0]
                                    mark_attendance(student_id, period_id, connection, is_within_time_limit=True)
                                    recognized_students.add(student_id)
                                    logging.debug(f"Recognized and marked student ID {student_id} for period {period_id}")
                                else:
                                    logging.warning(f"Student '{name}' (queried as '{name_for_query}') not found in database.")
                                cursor.close()

                        except Exception as e:
                            logging.error(f"Error recognizing face: {e}")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow('Mark Attendance', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info(f"User exited attendance marking after {elapsed_time / 60:.1f} minutes.")
            break

    # Mark all remaining students as absent
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT student_id FROM Students")
        all_students = {row[0] for row in cursor.fetchall()}
        unmarked_students = all_students - recognized_students

        if unmarked_students:
            for student_id in unmarked_students:
                cursor.execute(
                    """
                    INSERT INTO Attendance (student_id, period_id, status, recorded_timestamp)
                    VALUES (%s, %s, 'absent', NOW())
                    ON CONFLICT (student_id, period_id) DO NOTHING
                    """,
                    (student_id, period_id)
                )
            connection.commit()
            logging.info(f"Marked {len(unmarked_students)} students as absent for period {period_id}")
            absent_message = f"Marked {len(unmarked_students)} students as absent."
        else:
            logging.info("No students to mark as absent for period {period_id}")
            absent_message = "No students to mark as absent."
    except Exception as e:
        logging.error(f"Error marking absent students: {e}")
        connection.rollback()
        cap.release()
        cv2.destroyAllWindows()
        connection.close()
        return jsonify({"status": "error", "message": f"Error marking absent students: {str(e)}"}), 500
    finally:
        cursor.close()

    cap.release()
    cv2.destroyAllWindows()
    connection.close()

    # Return JSON response instead of redirect
    return jsonify({
        "status": "success",
        "message": "Attendance marking completed.",
        "absent_message": absent_message,
        "recognized_count": len(recognized_students)
    })


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture', methods=['GET', 'POST'])
def capture():
    if request.method == 'POST':
        name = request.form['name']
        rollno = request.form['rollno']

        if len(rollno) != 6:
            flash("Roll No must be exactly 6 characters long.", "error")
            return redirect(url_for('capture'))

        exists, _ = check_rollno_in_aligned(rollno, ALIGNED_DATA_PATH)
        if exists:
            flash(f"Roll No {rollno} already exists in aligned dataset.", "error")
            return redirect(url_for('capture'))

        name = name.replace(" ", "_")
        folder_name = f"{rollno}-{name}"
        output_dir = os.path.join(RAW_DATA_PATH, folder_name)

        if os.path.exists(output_dir):
            flash(f"The folder for {name} with Roll No {rollno} already exists in raw dataset.", "error")
            return redirect(url_for('capture'))

        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            flash("Error: Could not open camera.", "error")
            return redirect(url_for('capture'))

        num_frames = 100
        duration = 10
        frame_interval = duration / num_frames
        frame_count = 0

        while frame_count < num_frames:
            ret, frame = cap.read()
            if not ret:
                flash("Error: Could not read frame.", "error")
                break
            frame_filename = os.path.join(output_dir, f'frame_{frame_count:03d}.jpg')
            cv2.imwrite(frame_filename, frame)
            frame_count += 1
            time.sleep(frame_interval)

        cap.release()
        flash(f"Finished capturing frames for Roll No {rollno} and Name {name}.", "success")
        return redirect(url_for('capture'))

    return render_template('capture.html')

@app.route('/align', methods=['GET', 'POST'])
def align():
    if request.method == 'POST':
        result = align_images(config)
        flash(f"Alignment completed. Total images: {result['total']}, Successfully aligned: {result['aligned']}", "success")
        print(result['total'])
        return redirect(url_for('align'))
    return render_template('align.html')

@app.route('/populate', methods=['GET', 'POST'])
def populate():
    if request.method == 'POST':
        result = populate_students()
        flash(f"Population completed. {result['new_users']} new students added.", "success")
        return redirect(url_for('populate'))
    return render_template('populate.html')

@app.route('/course', methods=['GET', 'POST'])
def course():
    connection = get_db_connection()
    courses = []
    if connection:
        cursor = connection.cursor()
        cursor.execute("SELECT course_id, course_name FROM Courses ORDER BY course_name")
        courses = cursor.fetchall()
        cursor.close()
        connection.close()

    if request.method == 'POST':
        course_option = request.form['course_option']
        start_time = request.form['start_time']
        period_date = request.form['period_date']
        duration = request.form.get('duration', '').strip()  # Get duration from form, default to empty string

        # Validate duration
        try:
            duration = int(duration)
            if duration < 45 or duration > 120:
                flash("Duration must be between 45 and 90 minutes.", "error")
                return redirect(url_for('course'))
        except ValueError:
            flash("Duration must be a valid integer.", "error")
            return redirect(url_for('course'))

        connection = get_db_connection()
        if not connection:
            flash("Error connecting to database.", "error")
            return redirect(url_for('course'))

        cursor = connection.cursor()
        try:
            if course_option == 'new':
                course_name = request.form['course_name']
                if not course_name:
                    flash("Course name is required when adding a new course.", "error")
                    return redirect(url_for('course'))

                cursor.execute(
                    "INSERT INTO Courses (course_name, course_code) VALUES (%s, %s) RETURNING course_id",
                    (course_name, f"{course_name[:3].upper()}101")
                )
                course_id = cursor.fetchone()[0]
                flash(f"New course {course_name} added successfully.", "success")
            else:
                course_id = course_option
                cursor.execute("SELECT course_name FROM Courses WHERE course_id = %s", (course_id,))
                course_name = cursor.fetchone()[0]

            # Insert the new class period with the custom duration
            cursor.execute(
                """
                INSERT INTO Class_Periods (course_id, period_date, start_time, duration)
                VALUES (%s, %s, %s, %s)
                """,
                (course_id, period_date, start_time, duration)
            )
            connection.commit()
            flash(f"Class period for {course_name} on {period_date} at {start_time} with duration {duration} minutes added successfully.", "success")

        except Exception as e:
            connection.rollback()
            flash(f"Error adding class period: {str(e)}", "error")
        finally:
            cursor.close()
            connection.close()

        return redirect(url_for('course'))

    return render_template('course.html', courses=courses)


@app.route('/get_attendance/<int:period_id>', methods=['GET'])
def get_attendance(period_id):
    connection = get_db_connection()
    records = []
    course_name = "Unknown Subject"

    if connection:
        cursor = connection.cursor()
        cursor.execute("""
            SELECT s.first_name, s.middle_name, s.last_name, s.rollno, a.status, a.recorded_timestamp
            FROM Attendance a
            JOIN Students s ON a.student_id = s.student_id
            WHERE a.period_id = %s
        """, (period_id,))
        records = cursor.fetchall()

        cursor.execute("""
            SELECT c.course_name
            FROM Class_Periods cp
            JOIN Courses c ON cp.course_id = c.course_id
            WHERE cp.period_id = %s
        """, (period_id,))
        course_name_result = cursor.fetchone()
        course_name = course_name_result[0] if course_name_result else "Unknown Subject"
        cursor.close()
        connection.close()

    # Convert records to a list of dictionaries for JSON
    attendance_data = {
        "course_name": course_name,
        "records": [
            {
                "first_name": r[0],
                "middle_name": r[1] if r[1] else "",
                "last_name": r[2],
                "roll_no": r[3],
                "status": r[4],
                # Format timestamp as "YYYY-MM-DD HH:MM AM/PM" (12-hour)
                "timestamp": r[5].strftime("%Y-%m-%d %I:%M %p") if r[5] else "N/A"
            }
            for r in records
        ]
    }
    return jsonify(attendance_data)

@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    periods = get_periods()
    selected_period_id = None
    records = None
    course_name = None

    if request.method == 'POST':
        # Handle any unexpected POST requests (e.g., if AJAX fails)
        period_id = request.form.get('period_id')
        if period_id:
            selected_period_id = period_id

    if request.args.get('period_id') or selected_period_id:
        selected_period_id = selected_period_id or request.args.get('period_id')
        connection = get_db_connection()
        if connection:
            cursor = connection.cursor()
            cursor.execute("""
                SELECT c.course_name
                FROM Class_Periods cp
                JOIN Courses c ON cp.course_id = c.course_id
                WHERE cp.period_id = %s
            """, (selected_period_id,))
            course_name_result = cursor.fetchone()
            course_name = course_name_result[0] if course_name_result else "Unknown Subject"
            logging.debug(f"Course name for period {selected_period_id}: {course_name}")
            cursor.close()
            connection.close()

    return render_template('attendance.html', periods=periods, selected_period_id=selected_period_id, course_name=course_name)


if __name__ == '__main__':
    app.run(debug=True)