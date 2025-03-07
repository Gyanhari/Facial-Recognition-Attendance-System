import os
import random
import numpy as np
import torch
from PIL import Image
import imageio
from facenet_pytorch import MTCNN, InceptionResnetV1
from concurrent.futures import ThreadPoolExecutor, as_completed
from .helper import get_dataset
from tqdm import tqdm
import cv2

# Initialize MTCNN model from facenet-pytorch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709)
inception_model = InceptionResnetV1(pretrained='vggface2').eval()
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

def check_rollno_in_aligned(rollno, aligned_dir):
    try:
        aligned_folders = [f for f in os.listdir(aligned_dir) if os.path.isdir(os.path.join(aligned_dir, f))]
        for folder in aligned_folders:
            if folder.startswith(f"{rollno}-"):
                return True, folder
        return False, None
    except FileNotFoundError:
        print(f"Warning: Directory {aligned_dir} not found. Assuming no existing students.")
        return False, None
    except Exception as e:
        print(f"Error accessing directory {aligned_dir}: {e}")
        return False, None

def preprocess_image(img):
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    alpha = 1.5
    beta = 50
    img_adjusted = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)
    img_rgb = cv2.cvtColor(img_adjusted, cv2.COLOR_BGR2RGB)
    return img_rgb

def process_image(image_path, output_class_dir, failed_dir, margin, image_size, crop_size):
    filename = os.path.splitext(os.path.split(image_path)[1])[0]
    output_filename = os.path.join(output_class_dir, filename + '.png')
    failed_filename = os.path.join(failed_dir, filename + '.jpg')

    if os.path.exists(output_filename):
        return None

    try:
        img = imageio.imread(image_path)
    except (IOError, ValueError, IndexError) as e:
        print(f"Error reading {image_path}: {e}")
        return None

    if img.ndim < 2:
        print(f'Unable to align "{image_path}" (invalid dimensions)')
        return None

    if img.ndim == 2:
        img = np.stack((img,) * 3, axis=-1)
    img = img[:, :, 0:3]

    img = preprocess_image(img)

    boxes, _ = mtcnn.detect(img)
    if boxes is None:
        os.makedirs(failed_dir, exist_ok=True)
        Image.fromarray(img).save(failed_filename)
        return None

    img_size = np.asarray(img.shape)[0:2]
    results = []

    det = np.squeeze(boxes[0]) if boxes.size > 0 else None
    if det is not None:
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])

        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]

        scaled = np.array(Image.fromarray(cropped).resize((image_size, image_size), Image.BILINEAR))

        start_x = (image_size - crop_size) // 2
        start_y = (image_size - crop_size) // 2
        cropped_final = scaled[start_y:start_y + crop_size, start_x:start_x + crop_size, :]

        cropped_final = cropped_final / 255.0

        output_filename_n = output_filename
        Image.fromarray((cropped_final * 255).astype(np.uint8)).save(output_filename_n)
        results.append(f'{output_filename_n} {bb[0]} {bb[1]} {bb[2]} {bb[3]}')

    return results if results else None

def align_images(config):
    input_dir = os.path.expanduser(config["input_dir"])
    output_dir = os.path.expanduser(config["output_dir"])
    failed_dir = os.path.expanduser(config["failed_dir"])
    image_size = config["image_size"]
    crop_size = config["crop_size"]
    margin = config["margin"]
    random_order = config["random_order"]

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)

    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(output_dir, f'bounding_boxes_{random_key:05d}.txt')

    dataset = get_dataset(input_dir)
    if random_order:
        random.shuffle(dataset)

    nrof_images_total = 0
    nrof_successfully_aligned = 0
    bounding_boxes_data = []

    total_images = sum(len(cls.image_paths) for cls in dataset if not check_rollno_in_aligned(cls.name.split('-')[0], output_dir)[0])

    with ThreadPoolExecutor() as executor:
        futures = []
        for cls in dataset:
            try:
                rollno = cls.name.split('-')[0]
                exists, existing_folder = check_rollno_in_aligned(rollno, output_dir)
                if exists:
                    print(f"Skipping class {cls.name} (Roll No {rollno}) as it already exists in aligned dataset as {existing_folder}")
                    continue

                output_class_dir = os.path.join(output_dir, cls.name)
                failed_class_dir = os.path.join(failed_dir, cls.name)
                os.makedirs(output_class_dir, exist_ok=True)
                os.makedirs(failed_class_dir, exist_ok=True)

                if random_order:
                    random.shuffle(cls.image_paths)

                for image_path in cls.image_paths:
                    nrof_images_total += 1
                    futures.append(executor.submit(process_image, image_path, output_class_dir, failed_class_dir, margin, image_size, crop_size))

            except IndexError:
                print(f"Skipping invalid class name format: {cls.name}")
                continue

        with tqdm(total=total_images, desc="Aligning Images") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    nrof_successfully_aligned += 1
                    bounding_boxes_data.extend(result)
                pbar.update(1)

    with open(bounding_boxes_filename, "w") as text_file:
        text_file.write("\n".join(bounding_boxes_data))

    print(f'Total number of images: {nrof_images_total}')
    print(f'Number of successfully aligned images: {nrof_successfully_aligned}')

    return {
        'total': nrof_images_total,
        'aligned': nrof_successfully_aligned
    }