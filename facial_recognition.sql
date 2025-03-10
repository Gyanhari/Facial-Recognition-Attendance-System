-- Create database (optional, assuming a database is already selected)
-- CREATE DATABASE attendance_db;
-- USE attendance_db;

-- Disable foreign key checks temporarily to avoid order issues during creation
SET FOREIGN_KEY_CHECKS = 0;

-- Table: courses
CREATE TABLE courses (
    course_id INT NOT NULL AUTO_INCREMENT,
    course_name VARCHAR(100) NOT NULL,
    course_code VARCHAR(20) NOT NULL,
    PRIMARY KEY (course_id),
    UNIQUE KEY courses_course_code_key (course_code)
) ENGINE=InnoDB;

-- Table: students
CREATE TABLE students (
    student_id INT NOT NULL AUTO_INCREMENT,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    rollno VARCHAR(20) NOT NULL,
    middle_name VARCHAR(50),
    PRIMARY KEY (student_id),
    UNIQUE KEY students_enrollment_number_key (rollno)
) ENGINE=InnoDB;

-- Table: class_periods
CREATE TABLE class_periods (
    period_id INT NOT NULL AUTO_INCREMENT,
    course_id INT NOT NULL,
    period_date DATE NOT NULL,
    start_time TIME NOT NULL,
    duration INT NOT NULL DEFAULT 60,
    PRIMARY KEY (period_id),
    CONSTRAINT class_periods_duration_check CHECK (duration >= 45 AND duration <= 120),
    CONSTRAINT class_periods_course_id_fkey FOREIGN KEY (course_id) 
        REFERENCES courses (course_id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- Table: attendance
CREATE TABLE attendance (
    attendance_id INT NOT NULL AUTO_INCREMENT,
    student_id INT NOT NULL,
    period_id INT NOT NULL,
    status VARCHAR(10),
    recorded_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (attendance_id),
    UNIQUE KEY unique_student_period (student_id, period_id),
    CONSTRAINT attendance_status_check CHECK (status IN ('present', 'absent', 'late')),
    CONSTRAINT attendance_student_id_fkey FOREIGN KEY (student_id) 
        REFERENCES students (student_id) ON DELETE CASCADE,
    CONSTRAINT attendance_period_id_fkey FOREIGN KEY (period_id) 
        REFERENCES class_periods (period_id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- Indexes
CREATE INDEX idx_attendance_student_period ON attendance (student_id, period_id);
CREATE INDEX idx_class_periods_date_time ON class_periods (period_date, start_time);
CREATE INDEX idx_students_enrollment ON students (rollno);

-- Re-enable foreign key checks
SET FOREIGN_KEY_CHECKS = 1;

-- Sequence-like behavior is handled by AUTO_INCREMENT; no separate sequence tables needed
-- Initial sequence values from PostgreSQL (if needed, adjust AUTO_INCREMENT manually)
ALTER TABLE attendance AUTO_INCREMENT = 453; -- Next value after 452
ALTER TABLE class_periods AUTO_INCREMENT = 14; -- Next value after 13
ALTER TABLE courses AUTO_INCREMENT = 10; -- Next value after 9
ALTER TABLE students AUTO_INCREMENT = 88; -- Next value after 87

-- No data to insert (empty in original dump)
-- If needed later: INSERT INTO table_name (...) VALUES (...);