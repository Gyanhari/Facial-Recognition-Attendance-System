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
from facial_recognition.src.helper import admin_required
from flask_bcrypt import Bcrypt
bcrypt = Bcrypt()

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
                cursor.execute("SELECT course_id, course_name, course_code, semester FROM Courses ORDER BY semester, course_name")
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
                    SELECT cp.period_id, c.course_name, cp.period_date, cp.start_time, cp.duration, c.semester
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

def get_all_teachers():
    connection = get_db_connection()
    teachers = []
    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT teacher_id, first_name, last_name, email, is_admin FROM Teachers ORDER BY last_name, first_name")
                teachers = cursor.fetchall()
        except Exception as e:
            logger.error(f"Error fetching teachers: {str(e)}")
        finally:
            connection.close()
    return teachers


# Admin Dashboard
@admin_bp.route('dashboard')
@admin_required
def admin_dashboard():
    connection = get_db_connection()
    students = get_all_students()
    courses = []
    periods = []
    teachers = get_all_teachers()
    if connection:
        try:
            with connection.cursor() as cursor:
                # Fetch courses with assigned teachers
                cursor.execute("""
                    SELECT c.course_id, c.course_name, c.course_code, c.semester,
                           STRING_AGG(t.first_name || ' ' || t.last_name, ', ') AS assigned_teachers
                    FROM Courses c
                    LEFT JOIN Course_Teacher ct ON c.course_id = ct.course_id
                    LEFT JOIN Teachers t ON ct.teacher_id = t.teacher_id
                    GROUP BY c.course_id, c.course_name, c.course_code, c.semester
                    ORDER BY c.semester, c.course_name
                """)
                courses = cursor.fetchall()

                # Fetch periods (already updated in previous response)
                cursor.execute("""
                    SELECT cp.period_id, c.course_name, cp.period_date, cp.start_time, cp.duration, cp.completed,
                           t.first_name, t.last_name
                    FROM Class_Periods cp
                    JOIN Courses c ON cp.course_id = c.course_id
                    JOIN Course_Teacher ct ON cp.course_id = ct.course_id
                    JOIN Teachers t ON ct.teacher_id = t.teacher_id
                    ORDER BY cp.period_date DESC, cp.start_time
                """)
                periods = cursor.fetchall()
        finally:
            connection.close()
    return render_template('admin/dashboard.html', students=students, courses=courses, periods=periods, teachers=teachers)


# Student Management
@admin_bp.route('students', methods=['GET', 'POST'])
@admin_required
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
@admin_bp.route('courses', methods=['GET', 'POST'])
@admin_required
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
        semester = request.form.get('semester')
        teacher_id = request.form.get('teacher_id')  # New field for teacher assignment

        if action in ['add', 'edit']:
            if not course_name or not semester or not teacher_id:
                flash("Course Name, Semester, and Teacher are required.", "error")
                logger.warning("Missing Course Name, Semester, or Teacher for course management.")
                return redirect(url_for('admin.manage_courses'))
            try:
                semester = int(semester)
                if semester < 1 or semester > 8:
                    flash("Semester must be between 1 and 8.", "error")
                    logger.warning(f"Invalid semester value: {semester}")
                    return redirect(url_for('admin.manage_courses'))
            except ValueError:
                flash("Semester must be a valid integer.", "error")
                logger.warning(f"Invalid semester input: {semester}")
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
                        "INSERT INTO Courses (course_name, course_code, semester) VALUES (%s, %s, %s) RETURNING course_id",
                        (course_name, course_code, semester)
                    )
                    course_id = cursor.fetchone()[0]
                    # Assign teacher to the course
                    cursor.execute(
                        "INSERT INTO Course_Teacher (course_id, teacher_id) VALUES (%s, %s)",
                        (course_id, teacher_id)
                    )
                    connection.commit()
                    flash(f"Course {course_name} (Semester {semester}) added and assigned to teacher.", "success")
                    logger.info(f"Added course: {course_name} (ID: {course_id}, Semester: {semester}, Teacher ID: {teacher_id})")

                elif action == 'edit':
                    course_id = request.form.get('course_id')
                    if not course_id:
                        flash("Course ID is required for editing.", "error")
                        logger.warning("Missing Course ID for editing.")
                        return redirect(url_for('admin.manage_courses'))
                    course_name = course_name.strip()
                    course_code = course_code.strip() if course_code else f"{course_name[:3].upper()}101"
                    cursor.execute(
                        "UPDATE Courses SET course_name = %s, course_code = %s, semester = %s WHERE course_id = %s",
                        (course_name, course_code, semester, course_id)
                    )
                    # Update teacher assignment (delete existing, insert new)
                    cursor.execute("DELETE FROM Course_Teacher WHERE course_id = %s", (course_id,))
                    cursor.execute(
                        "INSERT INTO Course_Teacher (course_id, teacher_id) VALUES (%s, %s)",
                        (course_id, teacher_id)
                    )
                    connection.commit()
                    flash("Course updated successfully.", "success")
                    logger.info(f"Updated course ID {course_id}: {course_name} (Semester: {semester}, Teacher ID: {teacher_id})")

                elif action == 'delete':
                    course_id = request.form.get('course_id')
                    if not course_id:
                        flash("Course ID is required for deletion.", "error")
                        logger.warning("Missing Course ID for deletion.")
                        return redirect(url_for('admin.manage_courses'))
                    # Delete from Course_Teacher first due to foreign key constraint
                    cursor.execute("DELETE FROM Course_Teacher WHERE course_id = %s", (course_id,))
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

    connection = get_db_connection()
    courses = []
    teachers = []
    if connection:
        try:
            with connection.cursor() as cursor:
                # Fetch courses with assigned teacher info
                cursor.execute("""
                    SELECT c.course_id, c.course_name, c.course_code, c.semester,
                           t.first_name, t.last_name, t.email, t.teacher_id
                    FROM Courses c
                    LEFT JOIN Course_Teacher ct ON c.course_id = ct.course_id
                    LEFT JOIN Teachers t ON ct.teacher_id = t.teacher_id
                    ORDER BY c.semester, c.course_name
                """)
                courses = cursor.fetchall()

                # Fetch all teachers for the dropdown
                cursor.execute("SELECT teacher_id, first_name, last_name, email FROM Teachers ORDER BY last_name, first_name")
                teachers = cursor.fetchall()
        except Exception as e:
            logger.error(f"Error fetching courses or teachers: {str(e)}")
            flash("Error loading course data.", "error")
        finally:
            connection.close()

    return render_template('admin/courses.html', courses=courses, teachers=teachers)



# Class Period Management
@admin_bp.route('periods', methods=['GET', 'POST'])
@admin_required
def manage_periods():
    connection = get_db_connection()
    periods = []
    courses = []
    filter_completed = request.args.get('filter_completed', 'all')
    filter_date = request.args.get('filter_date', '')

    if connection:
        try:
            with connection.cursor() as cursor:
                # Fetch all courses for the form
                cursor.execute("""
                    SELECT course_id, course_name, course_code, semester
                    FROM Courses
                    ORDER BY semester, course_name
                """)
                courses = cursor.fetchall()

                # Fetch periods with completed status and course_id
                query = """
                    SELECT cp.period_id, c.course_name, cp.period_date, cp.start_time, cp.duration, c.semester, cp.completed, cp.course_id
                    FROM Class_Periods cp
                    JOIN Courses c ON cp.course_id = c.course_id
                    ORDER BY cp.period_date DESC, cp.start_time
                """
                if filter_completed == 'completed':
                    query = query.replace("ORDER BY", "WHERE cp.completed = TRUE ORDER BY")
                elif filter_completed == 'not_completed':
                    query = query.replace("ORDER BY", "WHERE cp.completed = FALSE ORDER BY")

                if filter_date:
                    if "WHERE" in query:
                        query = query.replace("ORDER BY", f"AND cp.period_date = '{filter_date}' ORDER BY")
                    else:
                        query = query.replace("ORDER BY", f"WHERE cp.period_date = '{filter_date}' ORDER BY")

                cursor.execute(query)
                periods = cursor.fetchall()

                # Handle form submissions
                if request.method == 'POST':
                    action = request.form.get('action')

                    if action in ['add', 'edit']:
                        course_id = request.form.get('course_id')

                        # Validate course_id is not empty and is an integer
                        if not course_id or not course_id.isdigit():
                            flash("Error: Invalid course selection.", "error")
                            return redirect(url_for('admin.manage_periods', filter_completed=filter_completed, filter_date=filter_date))

                        course_id = int(course_id)  # Convert to integer

                        # Validate course_id exists in Courses table
                        cursor.execute("SELECT course_id FROM Courses WHERE course_id = %s", (course_id,))
                        if not cursor.fetchone():
                            flash(f"Error: Course with ID {course_id} does not exist.", "error")
                            return redirect(url_for('admin.manage_periods', filter_completed=filter_completed, filter_date=filter_date))

                    if action == 'add':
                        period_date = request.form.get('period_date')
                        start_time = request.form.get('start_time')
                        duration = request.form.get('duration')
                        completed = request.form.get('completed') == 'on'

                        cursor.execute(
                            """
                            INSERT INTO Class_Periods (course_id, period_date, start_time, duration, completed)
                            VALUES (%s, %s, %s, %s, %s)
                            """,
                            (course_id, period_date, start_time, duration, completed)
                        )
                        connection.commit()
                        flash("Period added successfully.", "success")

                    elif action == 'edit':
                        period_id = request.form.get('period_id')
                        period_date = request.form.get('period_date')
                        start_time = request.form.get('start_time')
                        duration = request.form.get('duration')
                        completed = request.form.get('completed') == 'on'

                        cursor.execute(
                            """
                            UPDATE Class_Periods
                            SET course_id = %s, period_date = %s, start_time = %s, duration = %s, completed = %s
                            WHERE period_id = %s
                            """,
                            (course_id, period_date, start_time, duration, completed, period_id)
                        )
                        connection.commit()
                        flash("Period updated successfully.", "success")

                    elif action == 'delete':
                        period_id = request.form.get('period_id')
                        cursor.execute("DELETE FROM Class_Periods WHERE period_id = %s", (period_id,))
                        connection.commit()
                        flash("Period deleted successfully.", "success")

                    return redirect(url_for('admin.manage_periods', filter_completed=filter_completed, filter_date=filter_date))

        except Exception as e:
            connection.rollback()
            flash(f"Error: {str(e)}", "error")
        finally:
            connection.close()

    return render_template('admin/periods.html', periods=periods, courses=courses, filter_completed=filter_completed, filter_date=filter_date)


@admin_bp.route('attendance/<int:period_id>', methods=['GET', 'POST'])
def manage_attendance(period_id):
    connection = get_db_connection()
    records = []
    course_name = "Unknown Subject"
    if connection:
        try:
            with connection.cursor() as cursor:
                # Modify the query to format timestamp in AM/PM
                cursor.execute("""
                    SELECT s.first_name, s.middle_name, s.last_name, s.rollno, a.status, 
                           TO_CHAR(a.recorded_timestamp, 'YYYY-MM-DD HH12:MI:SS AM') AS recorded_timestamp
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
@admin_bp.route('monitor')
@admin_required
def monitor_system():
    log_file = os.path.join('logs', 'app.log')
    logs = []
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = f.readlines()[-20:]  # Last 20 log entries
    return render_template('admin/monitor.html', logs=logs, classifier_status=f"Loaded with {num_classes} classes" if classifier else "Not loaded")

# Re-train Model
@admin_bp.route('retrain', methods=['POST'])
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


@admin_bp.route('/add_teacher', methods=['GET', 'POST'])
@admin_required
def add_teacher():
    connection = get_db_connection()
    teachers = get_all_teachers()  # Fetch all teachers for display

    if request.method == 'POST':
        action = request.form.get('action')
        if not action:
            flash("Action is required.", "error")
            logger.warning("No action provided in teacher management form.")
            return redirect(url_for('admin.add_teacher'))

        connection = get_db_connection()
        if not connection:
            flash("Error connecting to database.", "error")
            logger.error("Failed to connect to database during teacher management.")
            return redirect(url_for('admin.add_teacher'))

        try:
            with connection.cursor() as cursor:
                if action == 'add':
                    first_name = request.form.get('first_name')
                    last_name = request.form.get('last_name')
                    email = request.form.get('email')
                    password = request.form.get('password')
                    is_admin = request.form.get('is_admin') == 'on'  # Changed to 'on' to match typical checkbox behavior

                    # Validate required fields
                    if not all([first_name, last_name, email, password]):
                        flash("All fields are required.", "error")
                        logger.warning("Missing required fields in add teacher form.")
                        return redirect(url_for('admin.add_teacher'))

                    if len(password) < 6:
                        flash("Password must be at least 6 characters long.", "error")
                        logger.warning("Password too short in add teacher form.")
                        return redirect(url_for('admin.add_teacher'))

                    # Check if email already exists
                    cursor.execute("SELECT teacher_id FROM Teachers WHERE email = %s", (email.lower().strip(),))
                    if cursor.fetchone():
                        flash("Email already exists.", "error")
                        logger.warning(f"Attempted to add teacher with existing email: {email}")
                        return redirect(url_for('admin.add_teacher'))

                    # Hash the password using bcrypt
                    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
                    
                    cursor.execute(
                        "INSERT INTO Teachers (first_name, last_name, email, password, is_admin) VALUES (%s, %s, %s, %s, %s) RETURNING teacher_id",
                        (first_name.strip(), last_name.strip(), email.lower().strip(), hashed_password, is_admin)
                    )
                    teacher_id = cursor.fetchone()[0]
                    connection.commit()
                    flash(f"Teacher {first_name} {last_name} added successfully.", "success")
                    logger.info(f"Added teacher: {first_name} {last_name} (ID: {teacher_id}, Email: {email}, Admin: {is_admin})")

                elif action == 'edit':
                    teacher_id = request.form.get('teacher_id')
                    first_name = request.form.get('first_name')
                    last_name = request.form.get('last_name')
                    email = request.form.get('email')
                    password = request.form.get('password')
                    is_admin = request.form.get('is_admin') == 'on'

                    if not all([teacher_id, first_name, last_name, email]):
                        flash("Teacher ID, First Name, Last Name, and Email are required.", "error")
                        logger.warning("Missing required fields in edit teacher form.")
                        return redirect(url_for('admin.add_teacher'))

                    # Check if email is taken by another teacher
                    cursor.execute(
                        "SELECT teacher_id FROM Teachers WHERE email = %s AND teacher_id != %s",
                        (email.lower().strip(), teacher_id)
                    )
                    if cursor.fetchone():
                        flash("Email is already in use by another teacher.", "error")
                        logger.warning(f"Email {email} already in use during teacher edit.")
                        return redirect(url_for('admin.add_teacher'))

                    if password:
                        if len(password) < 6:
                            flash("Password must be at least 6 characters long.", "error")
                            logger.warning("Password too short in edit teacher form.")
                            return redirect(url_for('admin.add_teacher'))
                        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
                        cursor.execute(
                            "UPDATE Teachers SET first_name = %s, last_name = %s, email = %s, password = %s, is_admin = %s WHERE teacher_id = %s",
                            (first_name.strip(), last_name.strip(), email.lower().strip(), hashed_password, is_admin, teacher_id)
                        )
                    else:
                        cursor.execute(
                            "UPDATE Teachers SET first_name = %s, last_name = %s, email = %s, is_admin = %s WHERE teacher_id = %s",
                            (first_name.strip(), last_name.strip(), email.lower().strip(), is_admin, teacher_id)
                        )
                    connection.commit()
                    flash("Teacher updated successfully.", "success")
                    logger.info(f"Updated teacher ID {teacher_id}: {first_name} {last_name} (Email: {email}, Admin: {is_admin})")

                elif action == 'delete':
                    teacher_id = request.form.get('teacher_id')
                    if not teacher_id:
                        flash("Teacher ID is required for deletion.", "error")
                        logger.warning("Missing teacher ID for deletion.")
                        return redirect(url_for('admin.add_teacher'))

                    # Check if teacher is assigned to any courses
                    cursor.execute("SELECT course_id FROM Course_Teacher WHERE teacher_id = %s", (teacher_id,))
                    if cursor.fetchone():
                        flash("Cannot delete teacher assigned to courses. Remove course assignments first.", "error")
                        logger.warning(f"Attempted to delete teacher ID {teacher_id} with existing course assignments.")
                        return redirect(url_for('admin.add_teacher'))

                    cursor.execute("DELETE FROM Teachers WHERE teacher_id = %s", (teacher_id,))
                    connection.commit()
                    flash("Teacher deleted successfully.", "success")
                    logger.info(f"Deleted teacher ID {teacher_id}")

        except Exception as e:
            connection.rollback()
            flash(f"Error: {str(e)}", "error")
            logger.error(f"Error during teacher management: {str(e)}")
        finally:
            connection.close()

        return redirect(url_for('admin.add_teacher'))

    return render_template('admin/add_teacher.html', teachers=teachers)
