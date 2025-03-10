import cv2
import time
import os

# Define the number of frames and duration
num_frames = 100
duration = 10
frame_interval = duration / num_frames

# Paths
RAW_DATA_PATH = os.path.join('facial_recognition', 'dataset', 'raw')
ALIGNED_DATA_PATH = os.path.join('facial_recognition', 'dataset', 'aligned')

def check_rollno_in_aligned(rollno):
    """Check if a folder with the given rollno exists in dataset/aligned"""
    try:
        aligned_folders = [f for f in os.listdir(ALIGNED_DATA_PATH) 
                          if os.path.isdir(os.path.join(ALIGNED_DATA_PATH, f))]
        for folder in aligned_folders:
            if folder.startswith(f"{rollno}-"):
                return True, folder
        return False, None
    except FileNotFoundError:
        print(f"Error: Directory {ALIGNED_DATA_PATH} not found. Assuming no existing students.")
        return False, None
    except Exception as e:
        print(f"Error accessing directory {ALIGNED_DATA_PATH}: {e}")
        return False, None

new_student = True
while new_student:
    name = input("Enter Person Name: ")
    rollno = input("Enter Student's six-character long Roll No: ")

    # Validate rollno length
    if len(rollno) != 6:
        print("Error: Roll No must be exactly six characters long. Please try again.")
        continue

    # Check if rollno already exists in dataset/aligned
    exists, existing_folder = check_rollno_in_aligned(rollno)
    if exists:
        print(f"Roll No {rollno} already exists in aligned dataset as {existing_folder}. Skipping...")
        again = input("Do you want to add another student (y/n): ")
        if again.lower() == "y":
            continue
        else:
            new_student = False
            break

    name = name.replace(" ", "_")
    print(f"Processed name: {name}")

    folder_name = f"{rollno}-{name}"
    output_dir = os.path.join(RAW_DATA_PATH, folder_name)

    # Check if the folder already exists in raw dataset
    if os.path.exists(output_dir):
        print(f"The folder for {name} with Roll No {rollno} already exists in raw dataset. Skipping...")
        again = input("Do you want to add another student (y/n): ")
        if again.lower() == "y":
            continue
        else:
            new_student = False
            break

    # Create the directory
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        break

    # Capture frames
    start_time = time.time()
    frame_count = 0

    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame_filename = os.path.join(output_dir, f'frame_{frame_count:03d}.jpg')
        cv2.imwrite(frame_filename, frame)
        print(f'Saved: {frame_filename}')

        frame_count += 1
        time.sleep(frame_interval)

    # Release the camera
    cap.release()
    print(f"Finished capturing frames for Roll No {rollno} and Name {name}.")

    again = input("Do you want to add another student (y/n): ")
    if again.lower() == "y":
        new_student = True
    else:
        new_student = False
        break