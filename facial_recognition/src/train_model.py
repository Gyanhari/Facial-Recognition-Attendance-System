import matplotlib
matplotlib.use('Agg')  # Use Agg backend (non-GUI)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pickle
from facenet_pytorch import InceptionResnetV1
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2  # Use OpenCV instead of PIL
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure all computations run on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    logging.warning("Warning: CUDA not available. Using CPU. Ensure NVIDIA drivers and PyTorch are properly installed.")
else:
    logging.info(f"Using device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# Load the FaceNet model
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Define dataset class
class FaceDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = int(self.labels[idx])
        return embedding, label

# Extract embeddings for each image using OpenCV
def get_embeddings(image_paths):
    embeddings = []
    for img_path in tqdm(image_paths, desc="Extracting embeddings", unit="image"):
        try:
            img = cv2.imread(img_path)
            if img is None:
                logging.error(f"Failed to load image: {img_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
            embedding = facenet(img_tensor)
            embeddings.append(embedding.detach().cpu().numpy().squeeze())
        except Exception as e:
            logging.error(f"Error processing image {img_path}: {e}")
    return embeddings

# Save embeddings and labels
def save_embeddings_and_labels(embeddings, labels, embeddings_dir):
    os.makedirs(embeddings_dir, exist_ok=True)
    np.save(os.path.join(embeddings_dir, 'embeddings.npy'), np.array(embeddings))
    np.save(os.path.join(embeddings_dir, 'labels.npy'), np.array(labels))

# Load embeddings and labels
def load_embeddings_and_labels(embeddings_dir):
    embeddings_path = os.path.join(embeddings_dir, 'embeddings.npy')
    labels_path = os.path.join(embeddings_dir, 'labels.npy')
    if os.path.exists(embeddings_path) and os.path.exists(labels_path):
        embeddings = np.load(embeddings_path, allow_pickle=True)
        labels = np.load(labels_path, allow_pickle=True)
        return embeddings, labels
    return np.array([]), np.array([])

# Update embeddings with new data
def update_embeddings_with_new_data(new_image_paths, new_labels, embeddings_dir):
    existing_embeddings, existing_labels = load_embeddings_and_labels(embeddings_dir)
    new_label_indices = [i for i, label in enumerate(new_labels) if label not in existing_labels]
    new_labels_to_add = [new_labels[i] for i in new_label_indices]
    new_image_paths_to_add = [new_image_paths[i] for i in new_label_indices]

    if not new_labels_to_add:
        logging.info("No new images found. Embeddings are up-to-date.")
        return

    logging.info(f"Found {len(new_labels_to_add)} new images. Extracting embeddings...")
    new_embeddings = get_embeddings(new_image_paths_to_add)
    updated_embeddings = np.vstack((existing_embeddings, new_embeddings)) if existing_embeddings.size > 0 else new_embeddings
    updated_labels = np.concatenate((existing_labels, new_labels_to_add)) if existing_labels.size > 0 else new_labels_to_add
    save_embeddings_and_labels(updated_embeddings, updated_labels, embeddings_dir)
    logging.info("Embeddings and labels updated successfully.")

# Define classifier model
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

# Train the classifier
def train_classifier(X_train, y_train, X_val, y_val, num_classes, epochs=50, batch_size=32, lr=0.001, patience=10, models_dir='facial_recognition/models', plots_dir='facial_recognition/plots'):
    classifier = Classifier(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

    train_dataset = FaceDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = FaceDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_loss_history = []
    val_loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        classifier.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for embeddings, labels in pbar:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = classifier(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix(loss=running_loss / (total / batch_size), accuracy=100 * correct / total)
            pbar.refresh()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        train_loss_history.append(epoch_loss)
        train_accuracy_history.append(epoch_accuracy)

        classifier.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings = embeddings.to(device)
                labels = labels.to(device)
                outputs = classifier(embeddings)
                val_loss = criterion(outputs, labels)
                val_running_loss += val_loss.item()
                _, val_predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (val_predicted == labels).sum().item()

        val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_accuracy)

        logging.info(
            f"Epoch [{epoch+1}/{epochs}] - "
            f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}% - "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(models_dir, exist_ok=True)
            torch.save(classifier.state_dict(), os.path.join(models_dir, 'best_classifier.pth'))
            logging.info(f"Saved best model checkpoint to {models_dir}.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info("Early stopping triggered.")
                break

    # Save loss and accuracy plots
    os.makedirs(plots_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Training Loss', marker='o')
    plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='Validation Loss', marker='o', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(plots_dir, 'loss_plot.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_accuracy_history) + 1), train_accuracy_history, label='Training Accuracy', marker='o')
    plt.plot(range(1, len(val_accuracy_history) + 1), val_accuracy_history, label='Validation Accuracy', marker='o', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(plots_dir, 'accuracy_plot.png'))
    plt.close()

    logging.info(f"Saved training plots to {plots_dir}.")
    return train_loss_history, val_loss_history, train_accuracy_history, val_accuracy_history

# Prepare data function
def prepare_training_data(aligned_dir, embeddings_dir):
    image_paths = []
    current_labels = []
    for subdir in os.listdir(aligned_dir):
        student_dir = os.path.join(aligned_dir, subdir)
        if os.path.isdir(student_dir):
            for img_name in os.listdir(student_dir):
                if img_name.endswith('.jpg') or img_name.endswith('.png'):
                    image_paths.append(os.path.join(student_dir, img_name))
                    current_labels.append(subdir)

    if os.path.exists(os.path.join(embeddings_dir, 'embeddings.npy')):
        logging.info("Loading embeddings and labels from file...")
        embeddings, saved_labels = load_embeddings_and_labels(embeddings_dir)
        update_embeddings_with_new_data(image_paths, current_labels, embeddings_dir)
        embeddings, current_labels = load_embeddings_and_labels(embeddings_dir)
    else:
        logging.info("Extracting embeddings from images...")
        embeddings = get_embeddings(image_paths)
        save_embeddings_and_labels(embeddings, current_labels, embeddings_dir)

    le = LabelEncoder()
    encoded_labels = le.fit_transform(current_labels)
    with open(os.path.join(embeddings_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)

    X_train, X_test, y_train, y_test = train_test_split(embeddings, encoded_labels, test_size=0.2, random_state=42)
    X_train = torch.tensor(np.array(X_train)).float().to(device)
    y_train = torch.tensor(np.array(y_train)).to(device)
    X_test = torch.tensor(np.array(X_test)).float().to(device)
    y_test = torch.tensor(np.array(y_test)).to(device)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    return X_train, y_train, X_val, y_val, len(np.unique(current_labels))