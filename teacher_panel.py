from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from flask_bcrypt import Bcrypt 
import psycopg2
from psycopg2 import Error
import logging

# Use the same logger as app.py
logger = logging.getLogger('facial_recognition_app')

# Blueprint for teacher panel
teacher_bp = Blueprint('teacher', __name__, template_folder='templates/teacher')

# Initialize Bcrypt (will be passed from app.py)
bcrypt = None  # We'll set this in app.py to avoid circular imports

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': '770312',
    'database': 'facial_recognition',
    'port': '5432'
}

# Database connection
def get_db_connection():
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Error as e:
        logger.error(f"Error connecting to database: {e}")
        return None

@teacher_bp.route('/')
def teacher_index():
    return render_template('teacher/base.html')

# Teacher Login
@teacher_bp.route('/login', methods=['GET', 'POST'])
def teacher_login():
    # Check if the user is already logged in
    if 'teacher_id' in session:
        flash("You are already logged in.", "info")
        return redirect(url_for('teacher.teacher_dashboard'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if not email or not password:
            flash("Email and password are required.", "error")
            return redirect(url_for('teacher.teacher_login'))

        connection = get_db_connection()
        if connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT teacher_id, first_name, last_name, password, is_admin FROM Teachers WHERE email = %s",
                        (email.lower().strip(),)
                    )
                    teacher = cursor.fetchone()
                    if teacher and bcrypt.check_password_hash(teacher[3], password):
                        session['teacher_id'] = teacher[0]
                        session['teacher_name'] = f"{teacher[1]} {teacher[2]}"
                        session['is_admin'] = teacher[4]
                        flash(f"Logged in as {session['teacher_name']}", "success")
                        return redirect(url_for('teacher.teacher_dashboard'))
                    else:
                        flash("Invalid email or password.", "error")
            finally:
                connection.close()
        return redirect(url_for('teacher.teacher_login'))
    return render_template('teacher/login.html')

# Teacher Registration
@teacher_bp.route('/register', methods=['GET', 'POST'])
def teacher_register():
    if request.method == 'POST':
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        email = request.form.get('email')
        password = request.form.get('password')

        if not all([first_name, last_name, email, password]):
            flash("All fields are required.", "error")
            return redirect(url_for('teacher.teacher_register'))

        # Hash the password
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        connection = get_db_connection()
        if connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1 FROM Teachers WHERE email = %s", (email,))
                    if cursor.fetchone():
                        flash("Email already registered.", "error")
                        return redirect(url_for('teacher.teacher_register'))

                    cursor.execute(
                        """
                        INSERT INTO Teachers (first_name, last_name, email, password)
                        VALUES (%s, %s, %s, %s) RETURNING teacher_id
                        """,
                        (first_name, last_name, email, hashed_password)
                    )
                    teacher_id = cursor.fetchone()[0]
                    connection.commit()
                    flash("Teacher registered successfully! Please log in.", "success")
                    logger.info(f"Registered teacher: {email} (ID: {teacher_id})")
                    return redirect(url_for('teacher.teacher_login'))
            except Exception as e:
                connection.rollback()
                logger.error(f"Error during teacher registration: {e}")
                flash(f"Error: {str(e)}", "error")
            finally:
                connection.close()
        else:
            flash("Error connecting to database.", "error")
        return redirect(url_for('teacher.teacher_register'))

    return render_template('teacher/register.html')

# Teacher Logout
@teacher_bp.route('/logout')
def teacher_logout():
    session.pop('teacher_id', None)
    session.pop('teacher_name', None)
    flash("Logged out successfully.", "success")
    return redirect(url_for('teacher.teacher_login'))

# Teacher Dashboard
@teacher_bp.route('/dashboard')
def teacher_dashboard():
    if 'teacher_id' not in session:
        flash("Please log in to access the dashboard.", "error")
        return redirect(url_for('teacher.teacher_login'))
    
    connection = get_db_connection()
    periods = []
    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT cp.period_id, c.course_name, cp.period_date, cp.start_time, cp.duration, c.semester, cp.completed
                    FROM Class_Periods cp
                    JOIN Courses c ON cp.course_id = c.course_id
                    JOIN Course_Teacher ct ON c.course_id = ct.course_id
                    WHERE ct.teacher_id = %s
                    ORDER BY cp.period_date DESC, cp.start_time
                """, (session['teacher_id'],))
                periods = cursor.fetchall()
        finally:
            connection.close()
    return render_template('teacher/dashboard.html', periods=periods)


def init_bcrypt(app):
    global bcrypt
    bcrypt = Bcrypt(app)