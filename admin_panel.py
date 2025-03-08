import os
import cv2
import time
import numpy as np
import torch
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
import psycopg2
from psycopg2 import Error
import pandas as pd
import logging
import pickle
from datetime import datetime
from facial_recognition.src.train_model import prepare_training_data, train_classifier, Classifier
from facial_recognition.src.helper import check_rollno_in_aligned

# Use the same logger as app.py
logger = logging.getLogger('facial_recognition_app')

# Blueprint for admin panel
admin_bp = Blueprint('admin', __name__, template_folder='templates/admin', static_folder='static')

# Database configuration (default, can be updated via settings)
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
EMBEDDINGS_DIR = os.path.join('facial_recognition', 'embeddings')
MODELS_DIR = os.path.join('facial_recognition', 'models')
PLOTS_DIR = os.path.join('facial_recognition', 'plots')

# Global classifier instance (shared with app.py if needed)
classifier = None
class_names = []
num_classes = 0

# Load initial classifier and class names
try:
    with open(os.path.join(EMBEDDINGS_DIR, 'label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)
    class_names = list(le.classes_)
    num_classes = len(class_names)
    classifier = Classifier(num_classes).to('cpu')
    classifier.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'best_classifier.pth'), map_location='cpu'))
    classifier.eval()
    logger.info(f"Admin panel loaded classifier with {num_classes} classes: {class_names}")
except Exception as e:
    logger.error(f"Error loading classifier in admin panel: {e}")

# Database connection
def get_db_connection():
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Error as e:
        logger.error(f"Error connecting to database: {e}")
        return None

# Helper function to fetch all students
def get_all_students():
    connection = get_db_connection()
    students = []
    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT student_id, rollno, first_name, middle_name, last_name FROM Students ORDER BY rollno")
                students = cursor.fetchall()
        except Exception as e:
            logger.error(f"Error fetching students: {e}")
        finally:
            connection.close()
    return students

# Helper function to fetch all courses
def get_all_courses():
    connection = get_db_connection()
    courses = []
    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT course_id, course_name, course_code FROM Courses ORDER BY course_name")
                courses = cursor.fetchall()
        except Exception as e:
            logger.error(f"Error fetching courses: {e}")
        finally:
            connection.close()
    return courses

# Helper function to fetch all class periods
def get_all_periods():
    connection = get_db_connection()
    periods = []
    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT cp.period_id, c.course_name, cp.period_date, cp.start_time, cp.duration
                    FROM Class_Periods cp
                    JOIN Courses c ON cp.course_id = c.course_id
                    ORDER BY cp.period_date, cp.start_time
                """)
                periods = cursor.fetchall()
        except Exception as e:
            logger.error(f"Error fetching periods: {e}")
        finally:
            connection.close()
    return periods

# Admin Dashboard
@admin_bp.route('/admin')
def admin_dashboard():
    students = get_all_students()
    courses = get_all_courses()
    periods = get_all_periods()
    return render_template('admin/dashboard.html', students=students, courses=courses, periods=periods)

# Student Management
@admin_bp.route('/admin/students', methods=['GET', 'POST'])
def manage_students():
    if request.method == 'POST':
        action = request.form.get('action')
        if not action:
            flash("Action is required.", "error")
            logger.warning("No action provided in student management form.")
            return redirect(url_for('admin.manage_students'))

        # Validate required fields for all actions
        rollno = request.form.get('rollno')
        first_name = request.form.get('first_name')
        middle_name = request.form.get('middle_name')  # Optional
        last_name = request.form.get('last_name')

        # Required fields check
        if action in ['add', 'edit']:
            if not rollno or not first_name or not last_name:
                flash("Roll No, First Name, and Last Name are required.", "error")
                logger.warning("Missing required fields for student management.")
                return redirect(url_for('admin.manage_students'))

        connection = get_db_connection()
        if not connection:
            flash("Error connecting to database.", "error")
            logger.error("Failed to connect to database during student management.")
            return redirect(url_for('admin.manage_students'))

        try:
            with connection.cursor() as cursor:
                if action == 'add':
                    rollno = rollno.strip()
                    if len(rollno) != 6 or not rollno.isdigit():
                        flash("Roll No must be exactly 6 digits.", "error")
                        logger.warning(f"Invalid Roll No {rollno} provided for adding student.")
                        return redirect(url_for('admin.manage_students'))
                    exists, _ = check_rollno_in_aligned(rollno, ALIGNED_DATA_PATH)
                    if exists:
                        flash(f"Roll No {rollno} already exists in aligned dataset.", "error")
                        logger.warning(f"Roll No {rollno} already exists in aligned dataset.")
                        return redirect(url_for('admin.manage_students'))
                    cursor.execute(
                        "INSERT INTO Students (rollno, first_name, middle_name, last_name) VALUES (%s, %s, %s, %s)",
                        (rollno, first_name.strip(), middle_name.strip() if middle_name else None, last_name.strip())
                    )
                    connection.commit()
                    flash(f"Student {first_name} {last_name} added successfully.", "success")
                    logger.info(f"Added student: {rollno} - {first_name} {last_name}")
                elif action == 'edit':
                    student_id = request.form.get('student_id')
                    if not student_id:
                        flash("Student ID is required for editing.", "error")
                        logger.warning("Missing student ID for editing.")
                        return redirect(url_for('admin.manage_students'))
                    rollno = rollno.strip()
                    cursor.execute(
                        "UPDATE Students SET rollno = %s, first_name = %s, middle_name = %s, last_name = %s WHERE student_id = %s",
                        (rollno, first_name.strip(), middle_name.strip() if middle_name else None, last_name.strip(), student_id)
                    )
                    connection.commit()
                    flash("Student updated successfully.", "success")
                    logger.info(f"Updated student ID {student_id}: {rollno} - {first_name} {last_name}")
                elif action == 'delete':
                    student_id = request.form.get('student_id')
                    if not student_id:
                        flash("Student ID is required for deletion.", "error")
                        logger.warning("Missing student ID for deletion.")
                        return redirect(url_for('admin.manage_students'))
                    cursor.execute("DELETE FROM Students WHERE student_id = %s", (student_id,))
                    connection.commit()
                    flash("Student deleted successfully.", "success")
                    logger.info(f"Deleted student ID {student_id}")
        except Exception as e:
            connection.rollback()
            flash(f"Error: {str(e)}", "error")
            logger.error(f"Error during student management: {str(e)}")
        finally:
            connection.close()

        return redirect(url_for('admin.manage_students'))

    students = get_all_students()
    return render_template('admin/students.html', students=students)


# Course Management
@admin_bp.route('/admin/courses', methods=['GET', 'POST'])
def manage_courses():
    if request.method == 'POST':
        action = request.form.get('action')
        if not action:
            flash("Action is required.", "error")
            logger.warning("No action provided in course management form.")
            return redirect(url_for('admin.manage_courses'))

        # Validate required fields
        course_name = request.form.get('course_name')
        course_code = request.form.get('course_code')  

        if action in ['add', 'edit'] and not course_name:
            flash("Course Name is required.", "error")
            logger.warning("Missing Course Name for course management.")
            return redirect(url_for('admin.manage_courses'))

        connection = get_db_connection()
        if not connection:
            flash("Error connecting to database.", "error")
            logger.error("Failed to connect to database during course management.")
            return redirect(url_for('admin.manage_courses'))

        try:
            with connection.cursor() as cursor:
                if action == 'add':
                    course_name = course_name.strip()
                    course_code = course_code.strip() if course_code else f"{course_name[:3].upper()}101"
                    cursor.execute(
                        "INSERT INTO Courses (course_name, course_code) VALUES (%s, %s) RETURNING course_id",
                        (course_name, course_code)
                    )
                    course_id = cursor.fetchone()[0]
                    connection.commit()
                    flash(f"Course {course_name} added successfully.", "success")
                    logger.info(f"Added course: {course_name} (ID: {course_id})")
                elif action == 'edit':
                    course_id = request.form.get('course_id')
                    if not course_id:
                        flash("Course ID is required for editing.", "error")
                        logger.warning("Missing Course ID for editing.")
                        return redirect(url_for('admin.manage_courses'))
                    course_name = course_name.strip()
                    course_code = course_code.strip() if course_code else f"{course_name[:3].upper()}101"
                    cursor.execute(
                        "UPDATE Courses SET course_name = %s, course_code = %s WHERE course_id = %s",
                        (course_name, course_code, course_id)
                    )
                    connection.commit()
                    flash("Course updated successfully.", "success")
                    logger.info(f"Updated course ID {course_id}: {course_name}")
                elif action == 'delete':
                    course_id = request.form.get('course_id')
                    if not course_id:
                        flash("Course ID is required for deletion.", "error")
                        logger.warning("Missing Course ID for deletion.")
                        return redirect(url_for('admin.manage_courses'))
                    cursor.execute("DELETE FROM Courses WHERE course_id = %s", (course_id,))
                    connection.commit()
                    flash("Course deleted successfully.", "success")
                    logger.info(f"Deleted course ID {course_id}")
        except Exception as e:
            connection.rollback()
            flash(f"Error: {str(e)}", "error")
            logger.error(f"Error during course management: {str(e)}")
        finally:
            connection.close()

        return redirect(url_for('admin.manage_courses'))

    courses = get_all_courses()
    return render_template('admin/courses.html', courses=courses)

# Class Period Management
@admin_bp.route('/admin/periods', methods=['GET', 'POST'])
def manage_periods():
    if request.method == 'POST':
        action = request.form.get('action')
        if not action:
            flash("Action is required.", "error")
            logger.warning("No action provided in period management form.")
            return redirect(url_for('admin.manage_periods'))

        connection = get_db_connection()
        if not connection:
            flash("Error connecting to database.", "error")
            logger.error("Failed to connect to database during period management.")
            return redirect(url_for('admin.manage_periods'))

        try:
            with connection.cursor() as cursor:
                if action == 'add':
                    course_id = request.form.get('course_id')
                    period_date = request.form.get('period_date')
                    start_time = request.form.get('start_time')
                    duration = request.form.get('duration')

                    if not all([course_id, period_date, start_time, duration]):
                        flash("All fields are required for adding a period.", "error")
                        logger.warning("Missing required fields for adding a period.")
                        return redirect(url_for('admin.manage_periods'))

                    duration = duration.strip()
                    if not duration.isdigit() or int(duration) < 45 or int(duration) > 120:
                        flash("Duration must be a number between 45 and 120 minutes.", "error")
                        logger.warning(f"Invalid duration {duration} for period (must be 45-120 minutes).")
                        return redirect(url_for('admin.manage_periods'))
                    duration = int(duration)
                    cursor.execute(
                        "INSERT INTO Class_Periods (course_id, period_date, start_time, duration) VALUES (%s, %s, %s, %s)",
                        (course_id, period_date, start_time, duration)
                    )
                    connection.commit()
                    flash("Class period added successfully.", "success")
                    logger.info(f"Added class period: Course ID {course_id}, Date {period_date}, Start {start_time}, Duration {duration}")
                elif action == 'edit':
                    period_id = request.form.get('period_id')
                    course_id = request.form.get('course_id')
                    period_date = request.form.get('period_date')
                    start_time = request.form.get('start_time')
                    duration = request.form.get('duration')

                    if not all([period_id, course_id, period_date, start_time, duration]):
                        flash("All fields are required for editing a period.", "error")
                        logger.warning("Missing required fields for editing a period.")
                        return redirect(url_for('admin.manage_periods'))

                    duration = duration.strip()
                    if not duration.isdigit() or int(duration) < 45 or int(duration) > 120:
                        flash("Duration must be a number between 45 and 120 minutes.", "error")
                        logger.warning(f"Invalid duration {duration} for period (must be 45-120 minutes).")
                        return redirect(url_for('admin.manage_periods'))
                    duration = int(duration)
                    cursor.execute(
                        "UPDATE Class_Periods SET course_id = %s, period_date = %s, start_time = %s, duration = %s WHERE period_id = %s",
                        (course_id, period_date, start_time, duration, period_id)
                    )
                    connection.commit()
                    flash("Class period updated successfully.", "success")
                    logger.info(f"Updated class period ID {period_id}: Course ID {course_id}, Date {period_date}, Start {start_time}, Duration {duration}")
                elif action == 'delete':
                    period_id = request.form.get('period_id')
                    if not period_id:
                        flash("Period ID is required for deletion.", "error")
                        logger.warning("Missing period ID for deletion.")
                        return redirect(url_for('admin.manage_periods'))
                    cursor.execute("DELETE FROM Class_Periods WHERE period_id = %s", (period_id,))
                    connection.commit()
                    flash("Class period deleted successfully.", "success")
                    logger.info(f"Deleted class period ID {period_id}")
        except Exception as e:
            connection.rollback()
            flash(f"Error: {str(e)}", "error")
            logger.error(f"Error during period management: {str(e)}")
        finally:
            connection.close()

        return redirect(url_for('admin.manage_periods'))

    periods = get_all_periods()
    courses = get_all_courses()
    return render_template('admin/periods.html', periods=periods, courses=courses)


# Attendance Management
@admin_bp.route('/admin/attendance/<int:period_id>', methods=['GET', 'POST'])
def manage_attendance(period_id):
    connection = get_db_connection()
    records = []
    course_name = "Unknown Subject"
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
                    SELECT c.course_name
                    FROM Class_Periods cp
                    JOIN Courses c ON cp.course_id = c.course_id
                    WHERE cp.period_id = %s
                """, (period_id,))
                course_name_result = cursor.fetchone()
                course_name = course_name_result[0] if course_name_result else course_name
        except Exception as e:
            logger.error(f"Error fetching attendance records for period {period_id}: {e}")
            flash(f"Error fetching attendance records: {str(e)}", "error")
        finally:
            connection.close()

    if request.method == 'POST':
        action = request.form.get('action')
        if not action:
            flash("Action is required.", "error")
            logger.warning("No action provided in attendance management form.")
            return redirect(url_for('admin.manage_attendance', period_id=period_id))

        connection = get_db_connection()
        if not connection:
            flash("Error connecting to database.", "error")
            logger.error("Failed to connect to database during attendance management.")
            return redirect(url_for('admin.manage_attendance', period_id=period_id))

        try:
            with connection.cursor() as cursor:
                if action == 'mark':
                    student_id = request.form.get('student_id')
                    status = request.form.get('status')
                    if not student_id or not status:
                        flash("Student ID and status are required for marking attendance.", "error")
                        logger.warning("Missing Student ID or status for marking attendance.")
                        return redirect(url_for('admin.manage_attendance', period_id=period_id))
                    cursor.execute(
                        "INSERT INTO Attendance (student_id, period_id, status, recorded_timestamp) VALUES (%s, %s, %s, NOW()) ON CONFLICT (student_id, period_id) DO UPDATE SET status = %s, recorded_timestamp = NOW()",
                        (student_id, period_id, status, status)
                    )
                    connection.commit()
                    flash("Attendance marked successfully.", "success")
                    logger.info(f"Marked attendance for student ID {student_id} in period {period_id}: {status}")
                elif action == 'export':
                    if not records:
                        flash("No attendance records to export.", "error")
                        logger.warning(f"No attendance records to export for period {period_id}.")
                        return redirect(url_for('admin.manage_attendance', period_id=period_id))
                    output_dir = "attendance_records"
                    os.makedirs(output_dir, exist_ok=True)
                    df = pd.DataFrame(records, columns=['First Name', 'Middle Name', 'Last Name', 'Roll No', 'Status', 'Timestamp'])
                    export_path = os.path.join(output_dir, f'attendance_period_{period_id}.csv')
                    df.to_csv(export_path, index=False)
                    flash(f"Attendance exported to {export_path}", "success")
                    logger.info(f"Exported attendance for period {period_id} to {export_path}")
        except Exception as e:
            connection.rollback()
            flash(f"Error: {str(e)}", "error")
            logger.error(f"Error during attendance management for period {period_id}: {str(e)}")
        finally:
            connection.close()

        return redirect(url_for('admin.manage_attendance', period_id=period_id))

    students = get_all_students()
    return render_template('admin/attendance.html', records=records, period_id=period_id, course_name=course_name, students=students)


# System Monitoring
@admin_bp.route('/admin/monitor')
def monitor_system():
    log_file = os.path.join('logs','app.log')
    logs = []
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = f.readlines()[-20:]  # Last 20 log entries
    return render_template('admin/monitor.html', logs=logs, classifier_status=f"Loaded with {num_classes} classes" if classifier else "Not loaded")

# Re-train Model
@admin_bp.route('/admin/retrain', methods=['POST'])
def retrain_model():
    try:
        if not os.path.exists(ALIGNED_DATA_PATH) or not os.listdir(ALIGNED_DATA_PATH):
            flash("No aligned data available for training.", "error")
            logger.warning("No aligned data available for training.")
            return redirect(url_for('admin.monitor_system'))

        logger.info("Preparing training data for re-training...")
        X_train, y_train, X_val, y_val, num_classes = prepare_training_data(ALIGNED_DATA_PATH, EMBEDDINGS_DIR)

        logger.info("Starting model re-training...")
        train_loss_history, val_loss_history, train_accuracy_history, val_accuracy_history = train_classifier(
            X_train, y_train, X_val, y_val, num_classes, epochs=50, models_dir=MODELS_DIR, plots_dir=PLOTS_DIR
        )

        global classifier, class_names
        classifier = Classifier(num_classes).to('cpu')
        classifier.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'best_classifier.pth'), map_location='cpu'))
        classifier.eval()

        with open(os.path.join(EMBEDDINGS_DIR, 'label_encoder.pkl'), 'rb') as f:
            le = pickle.load(f)
        class_names = list(le.classes_)
        logger.info(f"Re-trained model with {num_classes} classes: {class_names}")

        flash("Model re-trained successfully.", "success")
    except Exception as e:
        logger.error(f"Error during re-training: {e}")
        flash(f"Error during re-training: {str(e)}", "error")

    return redirect(url_for('admin.monitor_system'))


# Settings
@admin_bp.route('/admin/settings', methods=['GET', 'POST'])
def manage_settings():
    global DB_CONFIG
    if request.method == 'POST':
        new_config = {
            'host': request.form.get('host', DB_CONFIG['host']),
            'user': request.form.get('user', DB_CONFIG['user']),
            'password': request.form.get('password', DB_CONFIG['password']),
            'database': request.form.get('database', DB_CONFIG['database']),
            'port': request.form.get('port', DB_CONFIG['port'])
        }

        # Test the new configuration before applying
        try:
            test_conn = psycopg2.connect(**new_config)
            test_conn.close()
            DB_CONFIG = new_config
            flash("Database configuration updated successfully.", "success")
            logger.info("Updated database configuration.")
        except Exception as e:
            flash(f"Error updating database configuration: {str(e)}", "error")
            logger.error(f"Error updating database configuration: {str(e)}")
        return redirect(url_for('admin.manage_settings'))

    capture_params = {'num_frames': 100, 'duration': 10} 
    training_params = {'epochs': 50, 'batch_size': 32, 'lr': 0.001, 'patience': 10}
    return render_template('admin/settings.html', db_config=DB_CONFIG, capture_params=capture_params, training_params=training_params)    





