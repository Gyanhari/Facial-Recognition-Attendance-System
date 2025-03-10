import os
import psycopg2
from psycopg2 import Error

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': '770312',
    'database': 'facial_recognition',
    'port': '5432'
}

# Path to the dataset/aligned directory
DATASET_PATH = 'facial_recognition/dataset/aligned'

def get_db_connection():
    """Establish database connection"""
    try:
        connection = psycopg2.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f"Error connecting to database: {e}")
        return None

def get_folders_from_path(path):
    """Get list of folder names from the specified path"""
    try:
        # List all directories in the specified path
        folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        return folders
    except FileNotFoundError:
        print(f"Error: Directory {path} not found.")
        return []
    except Exception as e:
        print(f"Error accessing directory {path}: {e}")
        return []

def check_rollno_in_database(rollno, connection):
    """Check if a student with the given rollno already exists in the database"""
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT 1 FROM Students WHERE rollno = %s", (rollno,))
        exists = cursor.fetchone() is not None
        cursor.close()
        return exists
    except Exception as e:
        print(f"Error checking rollno {rollno} in database: {e}")
        return False

def populate_students():
    folders = get_folders_from_path(DATASET_PATH)
    connection = get_db_connection()
    if not connection:
        print("Failed to connect to database.")
        return {'new_users': 0}

    cursor = connection.cursor()
    new_users_count = 0

    try:
        for folder in folders:
            try:
                rollno, full_name = folder.split('-')
                # Check if the rollno already exists in the database
                if check_rollno_in_database(rollno, connection):
                    print(f"Skipping already uploaded user: {full_name} (Roll: {rollno})")
                    continue

                name_parts = full_name.split('_')
                if len(name_parts) < 2:
                    print(f"Skipping invalid folder name format: {folder} (insufficient name parts)")
                    continue

                # Extract first, middle, and last names
                first_name = name_parts[0].capitalize()
                last_name = name_parts[-1].capitalize()
                # Middle name is everything between the first and last parts
                middle_name = ' '.join(name_parts[1:-1]).capitalize() if len(name_parts) > 2 else None

                query = """
                    INSERT INTO Students (first_name, middle_name, last_name, rollno)
                    VALUES (%s, %s, %s, %s)
                    RETURNING student_id
                """
                cursor.execute(query, (first_name, middle_name, last_name, rollno))
                student_id = cursor.fetchone()[0]
                new_users_count += 1
                print(f"Inserted student {first_name} {middle_name or ''} {last_name} with ID {student_id}")

            except ValueError:
                print(f"Skipping invalid folder name format: {folder}")
                continue
            except Exception as e:
                print(f"Error processing folder {folder}: {e}")
                continue

        connection.commit()
        print(f"All operations completed. {new_users_count} new students added to the database.")
        return {'new_users': new_users_count}

    except Exception as e:
        print(f"Database error: {e}")
        connection.rollback()
        return {'new_users': 0}
    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":
    populate_students()