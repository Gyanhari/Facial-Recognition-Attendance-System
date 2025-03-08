import os
import cv2
import time
import numpy as np
import torch
from PIL import Image
import imageio
from facenet_pytorch import MTCNN, InceptionResnetV1
from facial_recognition.src.helper import get_dataset, check_rollno_in_aligned
from facial_recognition.src.align_dataset import align_images
from facial_recognition.src.train_model import prepare_training_data, train_classifier, Classifier
from facial_recognition.src.populate_databse import populate_students
import psycopg2
from psycopg2 import Error
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import torch.nn as nn
import pandas as pd
import logging
import pickle
from admin_panel import admin_bp

logger = logging.getLogger('facial_recognition_app')
logger.setLevel(logging.INFO)

for handler in logger.handlers[:]:
    logger.removeHandler(handler)
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, 'app.log')
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.register_blueprint(admin_bp, url_prefix='/admin')

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
EMBEDDINGS_DIR = os.path.join('facial_recognition', 'embeddings')
MODELS_DIR = os.path.join('facial_recognition', 'models')
PLOTS_DIR = os.path.join('facial_recognition', 'plots')

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

# Global classifier instance (updated during training)
classifier = None
class_names = []  # Initialize as an empty list
num_classes = 0

# Load initial classifier and class names
try:
    with open(os.path.join(EMBEDDINGS_DIR, 'label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)
    class_names = list(le.classes_)  # Convert to list to avoid NumPy array issues
    num_classes = len(class_names)
    classifier = Classifier(num_classes).to(device)
    classifier.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'best_classifier.pth'), map_location=device))
    classifier.eval()
    logger.info(f"Loaded classifier with {num_classes} classes: {class_names}")
except Exception as e:
    logger.error(f"Error loading initial classifier or label encoder: {e}")

config = {
    "input_dir": RAW_DATA_PATH,
    "output_dir": ALIGNED_DATA_PATH,
    "failed_dir": FAILED_DATA_PATH,
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
        logger.error(f"Error connecting to database: {e}")
        return None

def load_uploaded_users():
    uploaded_users = set()
    if os.path.exists(UPLOADED_USERS_FILE):
        with open(UPLOADED_USERS_FILE, 'r') as file:
            uploaded_users = {line.strip() for line in file if line.strip()}
    return uploaded_users

def save_uploaded_user(rollno):
    with open(UPLOADED_USERS_FILE, 'a') as file:
        file.write(f"{rollno}\n")

def get_periods():
    connection = get_db_connection()
    periods = []
    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT cp.period_id, c.course_name, cp.period_date, cp.start_time
                    FROM Class_Periods cp
                    JOIN Courses c ON cp.course_id = c.course_id
                    ORDER BY cp.period_date, cp.start_time
                """)
                periods = cursor.fetchall()
        finally:
            connection.close()
    return periods

def recognize_face(face_tensor):
    global classifier
    if classifier is None:
        logger.error("Classifier not loaded.")
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
        with connection.cursor() as cursor:
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
                logger.info(f"Successfully marked student ID {student_id} as {status} for period {period_id}")
                flash(f"Marked student ID {student_id} as {status}", "success")
            else:
                logger.info(f"Student ID {student_id} already marked with status {existing_status[0]} for period {period_id}")
    except Exception as e:
        logger.error(f"Error marking attendance for student ID {student_id}: {e}")
        connection.rollback()

@app.route('/mark_attendance/<int:period_id>', methods=['POST'])
def mark_attendance_route(period_id):
    global classifier, class_names, num_classes
    if classifier is None or len(class_names) == 0 or num_classes == 0:
        logger.error("Facial recognition models not loaded or class names empty.")
        return jsonify({"status": "error", "message": "Facial recognition models not loaded or class names empty."}), 500

    connection = get_db_connection()
    if not connection:
        return jsonify({"status": "error", "message": "Error connecting to database."}), 500

    # Check if the period has ended
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT period_date, start_time, duration
            FROM Class_Periods
            WHERE period_id = %s
            """,
            (period_id,)
        )
        period_info = cursor.fetchone()

    if not period_info:
        connection.close()
        return jsonify({"status": "error", "message": "Invalid period_id"}), 404

    period_date, start_time, duration = period_info
    period_start = datetime.combine(period_date, start_time)
    period_end = period_start + timedelta(minutes=duration)
    current_time = datetime.now()

    if current_time > period_end:
        logger.info(f"Period {period_id} has ended at {period_end}. Marking all unmarked students as absent.")
        with connection.cursor() as cursor:
            try:
                cursor.execute("SELECT student_id FROM Students")
                all_students = {row[0] for row in cursor.fetchall()}
                cursor.execute("SELECT student_id FROM Attendance WHERE period_id = %s", (period_id,))
                marked_students = {row[0] for row in cursor.fetchall()}
                unmarked_students = all_students - marked_students

                if unmarked_students:
                    cursor.executemany(
                        """
                        INSERT INTO Attendance (student_id, period_id, status, recorded_timestamp)
                        VALUES (%s, %s, 'absent', NOW())
                        ON CONFLICT (student_id, period_id) DO NOTHING
                        """,
                        [(student_id, period_id) for student_id in unmarked_students]
                    )
                    connection.commit()
                    logger.info(f"Marked {len(unmarked_students)} students as absent for period {period_id}")
                    absent_message = f"Marked {len(unmarked_students)} students as absent due to period end."
                else:
                    logger.info(f"No unmarked students to mark as absent for period {period_id}")
                    absent_message = "No unmarked students to mark as absent."
            except Exception as e:
                logger.error(f"Error marking absent students: {e}")
                connection.rollback()
                connection.close()
                return jsonify({"status": "error", "message": f"Error marking absent students: {str(e)}"}), 500

        connection.close()
        return jsonify({
            "status": "success",
            "message": "Attendance marking completed (period ended).",
            "absent_message": absent_message,
            "recognized_count": 0
        })

    # Proceed with live attendance if period is active
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
            logger.info("5-minute time limit reached. Stopping webcam.")
            break

        ret, frame = cap.read()
        if not ret:
            logger.error("Error: Failed to capture frame.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(frame_rgb)

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = [int(coord) for coord in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                face = frame_rgb[y1:y2, x1:x2]
                if face.size == 0 or face.shape[0] <= 0 or face.shape[1] <= 0:
                    continue

                face_pil = Image.fromarray(face)
                try:
                    face_cropped = mtcnn(face_pil)
                    if face_cropped is not None and len(face_cropped) > 0:
                        name, prob = recognize_face(face_cropped[0])
                        label = f"{name} ({prob:.2f})" if name and prob > 0.7 else "Unknown"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        if name and prob > 0.7:
                            try:
                                rollno_part, name_part = name.split('-', 1)
                                name_parts = name_part.split('_')
                                if len(name_parts) < 2:
                                    raise ValueError("Invalid name format")
                                first_name = name_parts[0]
                                last_name = name_parts[-1]

                                with connection.cursor() as cursor:
                                    cursor.execute(
                                        """
                                        SELECT student_id
                                        FROM Students
                                        WHERE rollno = %s AND LOWER(first_name) = LOWER(%s) AND LOWER(last_name) = LOWER(%s)
                                        """,
                                        (rollno_part, first_name, last_name)
                                    )
                                    student = cursor.fetchone()
                                    if student and student[0] not in recognized_students:
                                        student_id = student[0]
                                        mark_attendance(student_id, period_id, connection, is_within_time_limit=True)
                                        recognized_students.add(student_id)
                                        logger.info(f"Recognized and marked student ID {student_id} for period {period_id}")
                            except Exception as e:
                                logger.error(f"Error parsing or querying student '{name}': {e}")
                except Exception as e:
                    logger.error(f"Error processing face with MTCNN: {e}")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            logger.warning("No faces detected in the current frame.")

        cv2.imshow('Mark Attendance', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info(f"User exited attendance marking after {elapsed_time / 60:.1f} minutes.")
            break

    # Mark all remaining students as absent after live attendance
    with connection.cursor() as cursor:
        try:
            cursor.execute("SELECT student_id FROM Students")
            all_students = {row[0] for row in cursor.fetchall()}
            unmarked_students = all_students - recognized_students

            if unmarked_students:
                cursor.executemany(
                    """
                    INSERT INTO Attendance (student_id, period_id, status, recorded_timestamp)
                    VALUES (%s, %s, 'absent', NOW())
                    ON CONFLICT (student_id, period_id) DO NOTHING
                    """,
                    [(student_id, period_id) for student_id in unmarked_students]
                )
                connection.commit()
                logger.info(f"Marked {len(unmarked_students)} students as absent for period {period_id}")
                absent_message = f"Marked {len(unmarked_students)} students as absent."
            else:
                logger.info(f"No students to mark as absent for period {period_id}")
                absent_message = "No students to mark as absent."
        except Exception as e:
            logger.error(f"Error marking absent students: {e}")
            connection.rollback()

    cap.release()
    cv2.destroyAllWindows()
    connection.close()

    return jsonify({
        "status": "success",
        "message": "Attendance marking completed.",
        "absent_message": absent_message,
        "recognized_count": len(recognized_students)
    })

@app.route('/')
def index():
    return redirect(url_for('capture'))


@app.route('/capture', methods=['GET', 'POST'])
def capture():
    if request.method == 'POST':
        name = request.form['name'].strip()
        rollno = request.form['rollno'].strip()

        if not rollno or len(rollno) != 6 or not rollno.isdigit():
            flash("Roll No must be exactly 6 digits.", "error")
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
            cv2.imshow('Capturing...', frame)
            frame_filename = os.path.join(output_dir, f'frame_{frame_count:03d}.jpg')
            cv2.imwrite(frame_filename, frame)
            frame_count += 1
            time.sleep(frame_interval)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        flash(f"Finished capturing {frame_count} frames for Roll No {rollno} and Name {name}.", "success")
        save_uploaded_user(rollno)
        return redirect(url_for('capture'))

    return render_template('capture.html')

@app.route('/align', methods=['GET', 'POST'])
def align():
    if request.method == 'POST':
        result = align_images(config)
        flash(f"Alignment completed. Total images: {result['total']}, Successfully aligned: {result['aligned']}", "success")
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
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT course_id, course_name FROM Courses ORDER BY course_name")
                courses = cursor.fetchall()
        finally:
            connection.close()

    if request.method == 'POST':
        course_option = request.form['course_option']
        start_time = request.form['start_time']
        period_date = request.form['period_date']
        duration = request.form.get('duration', '').strip()

        try:
            duration = int(duration)
            if duration < 45 or duration > 120:
                flash("Duration must be between 45 and 120 minutes.", "error")
                return redirect(url_for('course'))
        except ValueError:
            flash("Duration must be a valid integer.", "error")
            return redirect(url_for('course'))

        connection = get_db_connection()
        if not connection:
            flash("Error connecting to database.", "error")
            return redirect(url_for('course'))

        try:
            with connection.cursor() as cursor:
                if course_option == 'new':
                    course_name = request.form['course_name'].strip()
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
            connection.close()

        return redirect(url_for('course'))

    return render_template('course.html', courses=courses)

@app.route('/get_attendance/<int:period_id>', methods=['GET'])
def get_attendance(period_id):
    connection = get_db_connection()
    records = []
    course_name = "Unknown Subject"
    subject_date = None

    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT s.first_name, s.middle_name, s.last_name, s.rollno, a.status, a.recorded_timestamp
                    FROM Attendance a
                    JOIN Students s ON a.student_id = s.student_id
                    WHERE a.period_id = %s
                """, (period_id,))
                records = cursor.fetchall()

                cursor.execute("""
                    SELECT c.course_name, cp.period_date
                    FROM Class_Periods cp
                    JOIN Courses c ON cp.course_id = c.course_id
                    WHERE cp.period_id = %s
                """, (period_id,))
                course_info = cursor.fetchone()
                if course_info:
                    course_name, subject_date = course_info
        finally:
            connection.close()

    attendance_data = {
        "course_name": course_name,
        "date": subject_date.strftime("%Y-%m-d") if subject_date else "N/A",
        "records": [
            {
                "first_name": r[0],
                "middle_name": r[1] if r[1] else "",
                "last_name": r[2],
                "roll_no": r[3],
                "status": r[4],
                "timestamp": r[5].strftime("%Y-%m-%d %I:%M %p") if r[5] else "N/A"
            }
            for r in records
        ]
    }
    return jsonify(attendance_data)

@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    periods = get_periods()
    selected_period_id = request.form.get('period_id') or request.args.get('period_id')
    course_name = None

    if selected_period_id:
        connection = get_db_connection()
        if connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute("""
                        SELECT c.course_name
                        FROM Class_Periods cp
                        JOIN Courses c ON cp.course_id = c.course_id
                        WHERE cp.period_id = %s
                    """, (selected_period_id,))
                    course_name_result = cursor.fetchone()
                    course_name = course_name_result[0] if course_name_result else "Unknown Subject"
                    logger.debug(f"Course name for period {selected_period_id}: {course_name}")
            finally:
                connection.close()

    return render_template('attendance.html', periods=periods, selected_period_id=selected_period_id, course_name=course_name)

@app.route('/train', methods=['GET', 'POST'])
def train():
    global classifier, class_names, num_classes
    if request.method == 'POST':
        # Ensure directories exist
        for directory in [EMBEDDINGS_DIR, MODELS_DIR, PLOTS_DIR]:
            os.makedirs(directory, exist_ok=True)

        try:
            logger.info("Preparing training data...")
            X_train, y_train, X_val, y_val, num_classes = prepare_training_data(ALIGNED_DATA_PATH, EMBEDDINGS_DIR)

            logger.info("Starting model training...")
            train_loss_history, val_loss_history, train_accuracy_history, val_accuracy_history = train_classifier(
                X_train, y_train, X_val, y_val, num_classes, epochs=50, models_dir=MODELS_DIR, plots_dir=PLOTS_DIR
            )

            # Update global classifier and class names
            classifier = Classifier(num_classes).to(device)
            classifier.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'best_classifier.pth'), map_location=device))
            classifier.eval()

            # Load the updated label encoder to get class names
            with open(os.path.join(EMBEDDINGS_DIR, 'label_encoder.pkl'), 'rb') as f:
                le = pickle.load(f)
            class_names = list(le.classes_)
            logger.info(f"Updated class_names after training: {class_names}")

            flash(f"Training completed successfully. Model saved to {MODELS_DIR}. Plots generated in {PLOTS_DIR}.", "success")
        except Exception as e:
            logger.error(f"Error during training: {e}")
            flash(f"Error during training: {str(e)}", "error")

        return redirect(url_for('train'))

    return render_template('train.html')

if __name__ == '__main__':
    app.run(debug=True)