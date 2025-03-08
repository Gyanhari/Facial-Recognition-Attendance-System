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
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Ensure all computations run on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("Warning: CUDA not available. Using CPU. Ensure NVIDIA drivers and PyTorch are properly installed.")
else:
    print(f"Using device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# Load the FaceNet model (InceptionResnetV1)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)  

# Define dataset class to handle aligned images
class FaceDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = int(self.labels[idx])  # Ensure labels are integers
        return embedding, label

# Extract embeddings for each image
def get_embeddings(image_paths):
    embeddings = []
    # Use tqdm to show progress for each image being processed
    for img_path in tqdm(image_paths, desc="Extracting embeddings", unit="image"):
        img = Image.open(img_path).convert('RGB')
        img = transforms.ToTensor()(img).unsqueeze(0).to(device)  # Move tensor to GPU/CPU
        embedding = facenet(img)  # Extract embedding
        embeddings.append(embedding.detach().cpu().numpy().squeeze())  # Detach and move to CPU for storage
    return embeddings

# Save embeddings and labels as .npy files
def save_embeddings_and_labels(embeddings, labels, embeddings_dir):
    os.makedirs(embeddings_dir, exist_ok=True)  # Ensure directory exists
    np.save(os.path.join(embeddings_dir, 'embeddings.npy'), np.array(embeddings))
    np.save(os.path.join(embeddings_dir, 'labels.npy'), np.array(labels))

# Load embeddings and labels from .npy files
def load_embeddings_and_labels(embeddings_dir):
    embeddings_path = os.path.join(embeddings_dir, 'embeddings.npy')
    labels_path = os.path.join(embeddings_dir, 'labels.npy')

    if os.path.exists(embeddings_path) and os.path.exists(labels_path):
        embeddings = np.load(embeddings_path, allow_pickle=True)
        labels = np.load(labels_path, allow_pickle=True)
        return embeddings, labels
    else:
        return np.array([]), np.array([])  # Return empty numpy arrays

# Update embeddings with new data
def update_embeddings_with_new_data(new_image_paths, new_labels, embeddings_dir):
    # Load existing embeddings and labels
    existing_embeddings, existing_labels = load_embeddings_and_labels(embeddings_dir)

    # Identify new labels and their corresponding image paths
    new_label_indices = [i for i, label in enumerate(new_labels) if label not in existing_labels]
    new_labels_to_add = [new_labels[i] for i in new_label_indices]
    new_image_paths_to_add = [new_image_paths[i] for i in new_label_indices]

    if not new_labels_to_add:
        print("No new images found. Embeddings are up-to-date.")
        return

    # Extract embeddings only for new labels
    print(f"Found {len(new_labels_to_add)} new images. Extracting embeddings...")
    new_embeddings = get_embeddings(new_image_paths_to_add)

    # Append new embeddings and labels to existing data
    updated_embeddings = np.vstack((existing_embeddings, new_embeddings)) if existing_embeddings.size > 0 else new_embeddings
    updated_labels = np.concatenate((existing_labels, new_labels_to_add)) if existing_labels.size > 0 else new_labels_to_add

    # Save updated embeddings and labels
    save_embeddings_and_labels(updated_embeddings, updated_labels, embeddings_dir)
    print("Embeddings and labels updated successfully.")

aligned_dir = 'facial_recognition/dataset/aligned' 
embeddings_dir = 'facial_recognition/embeddings'

image_paths = []
current_labels = []

for subdir in os.listdir(aligned_dir):
    student_dir = os.path.join(aligned_dir, subdir)
    if os.path.isdir(student_dir):
        for img_name in os.listdir(student_dir):
            if img_name.endswith('.jpg') or img_name.endswith('.png'):
                image_paths.append(os.path.join(student_dir, img_name))
                current_labels.append(subdir)  # Current label is the subdirectory name (student name)

# Check if embeddings file exists
if os.path.exists(os.path.join(embeddings_dir, 'embeddings.npy')):
    print("Loading embeddings and labels from file...")
    embeddings, saved_labels = load_embeddings_and_labels(embeddings_dir)
    # Check for new labels and update embeddings if necessary
    update_embeddings_with_new_data(image_paths, current_labels, embeddings_dir)
    # Reload embeddings and labels after updating
    embeddings, current_labels = load_embeddings_and_labels(embeddings_dir)
else:
    print("Extracting embeddings from images...")
    embeddings = get_embeddings(image_paths)
    save_embeddings_and_labels(embeddings, current_labels, embeddings_dir)  # Save current_labels

# Encode labels
le = LabelEncoder()
encoded_labels = le.fit_transform(current_labels)

# Save the LabelEncoder for future use
with open(os.path.join(embeddings_dir, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(le, f)

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(embeddings, encoded_labels, test_size=0.2, random_state=42)

# Convert embeddings and labels to tensors and move to GPU
X_train = torch.tensor(np.array(X_train)).float().to(device)
y_train = torch.tensor(np.array(y_train)).to(device)
X_test = torch.tensor(np.array(X_test)).float().to(device)
y_test = torch.tensor(np.array(y_test)).to(device)

# Define a simple classifier model
class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(512, 128),  # FaceNet output size is 512
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)  # Number of classes (students)
        )

    def forward(self, x):
        return self.fc(x)

# Initialize and move the classifier model to GPU
num_classes = len(np.unique(current_labels))
classifier = Classifier(num_classes).to(device) 

# Train the classifier
def train(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, lr=0.001, patience=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
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
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for embeddings, labels in pbar:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Track training loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            pbar.set_postfix(loss=running_loss / (total / batch_size), accuracy=100 * correct / total)
            pbar.refresh()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        train_loss_history.append(epoch_loss)
        train_accuracy_history.append(epoch_accuracy)

        # Validation loop
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings = embeddings.to(device)
                labels = labels.to(device)

                outputs = model(embeddings)
                val_loss = criterion(outputs, labels)
                val_running_loss += val_loss.item()

                _, val_predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (val_predicted == labels).sum().item()

        val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_accuracy)

        print(
            f"Epoch [{epoch+1}/{epochs}] - "
            f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}% - "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
        )

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs('facial_recognition/models', exist_ok=True)  # Ensure directory exists
            torch.save(model.state_dict(), 'facial_recognition/models/best_classifier.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return train_loss_history, val_loss_history, train_accuracy_history, val_accuracy_history

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train the classifier and get the loss/accuracy history
train_loss_history, val_loss_history, train_accuracy_history, val_accuracy_history = train(
    classifier, X_train, y_train, X_val, y_val, epochs=50
)

# Plot Training and Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Training Loss', marker='o')
plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='Validation Loss', marker='o', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid()
plt.show()

# Plot Training and Validation Accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_accuracy_history) + 1), train_accuracy_history, label='Training Accuracy', marker='o')
plt.plot(range(1, len(val_accuracy_history) + 1), val_accuracy_history, label='Validation Accuracy', marker='o', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy Over Epochs')
plt.legend()
plt.grid()
plt.show()
