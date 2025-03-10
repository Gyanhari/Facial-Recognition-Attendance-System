import os
from functools import wraps
from flask import redirect, url_for, flash, session
from flask_bcrypt import Bcrypt  # Add this import


class ImageClass:
    """Helper class to store image class name and image paths."""
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

def get_image_paths(facedir):
    """Get paths to all image files in a directory."""
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_paths = []
    for root, dirs, files in os.walk(facedir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in supported_formats):
                image_paths.append(os.path.join(root, file))
    return image_paths

def get_dataset(path, has_class_directories=True):
    """Reads the dataset from the specified path."""
    dataset = []
    path_exp = os.path.expanduser(path)
    
    if has_class_directories:
        classes = [path for path in os.listdir(path_exp)
                   if os.path.isdir(os.path.join(path_exp, path))]
        classes.sort()
        nrof_classes = len(classes)

        for i in range(nrof_classes):
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            image_paths = get_image_paths(facedir)
            dataset.append(ImageClass(class_name, image_paths))
    else:
        image_paths = get_image_paths(path_exp)
        dataset.append(ImageClass('all_images', image_paths))

    return dataset

def check_rollno_in_aligned(rollno, aligned_dir):
    try:
        aligned_folders = [f for f in os.listdir(aligned_dir) if os.path.isdir(os.path.join(aligned_dir, f))]
        for folder in aligned_folders:
            if folder.startswith(f"{rollno}-"):
                return True, folder
        return False, None
    except FileNotFoundError:
        return False, None

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'teacher_id' not in session or not session.get('is_admin', False):
            flash("Admin access required.", "error")
            return redirect(url_for('teacher.teacher_login'))
        return f(*args, **kwargs)
    return decorated_function
