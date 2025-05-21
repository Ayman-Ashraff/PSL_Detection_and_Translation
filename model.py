import torch
import torch.nn as nn
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from pytorchvideo.models.resnet import create_resnet
import matplotlib.pyplot as plt

# Define functions
def extract_frames(video_path, num_frames=16, transform=None, resize_shape=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError("Video has no frames: " + video_path)

    frame_interval = max(1, total_frames // num_frames)

    for i in range(num_frames):
        frame_id = i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        success, frame = cap.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if transform:
                frame = transform(Image.fromarray(frame))
            else:
                frame = cv2.resize(frame, resize_shape)
                frame = torch.from_numpy(frame).permute(2, 0, 1).float()
            frames.append(frame)

    cap.release()

    if len(frames) < num_frames:
        if len(frames) == 0:
            raise ValueError("No frames extracted from video: " + video_path)
        last_frame = frames[-1]
        while len(frames) < num_frames:
            frames.append(last_frame)

    return torch.stack(frames)


class VideoDataset(Dataset):
    def _init_(self, root_dir, num_frames=16, transform=None):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform
        self.videos = []
        self.labels = []
        self.class_to_idx = {}
        self.subclass_to_idx = {}
        self.idx = 0  # Global index to assign label

        for cls in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, cls)
            if os.path.isdir(class_dir):  # Only process directories
                self.class_to_idx[cls] = self.idx
                self.idx += 1
                for sub_cls in os.listdir(class_dir):
                    subclass_dir = os.path.join(class_dir, sub_cls)
                    if os.path.isdir(subclass_dir):  # Only process directories
                        self.subclass_to_idx[sub_cls] = self.idx
                        for video_file in os.listdir(subclass_dir):
                            video_path = os.path.join(subclass_dir, video_file)
                            if os.path.isfile(video_path):  # Ensure it's a file
                                self.videos.append(video_path)
                                self.labels.append(self.idx)

        self.classes = list(self.class_to_idx.keys())  # List of all top-level classes

    def _len_(self):
        return len(self.videos)

    def _getitem_(self, idx):
        video_path = self.videos[idx]
        label = self.labels[idx]
        frames = extract_frames(video_path, self.num_frames, self.transform)
        return frames, label


# Modify this path to where your dataset is located
dataset_path = "/home/drfarooq/Desktop/Ayman/NDS"  # Change to your dataset directory
dataset = VideoDataset(dataset_path, num_frames=16, transform=None)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Load Pre-trained I3D Model (ResNet-50 backbone)
model = create_resnet(
    input_channel=3,        # RGB input
    model_depth=50,         # ResNet-50 backbone
    model_num_class=400,    # Pre-trained on Kinetics-400 dataset
    norm=nn.BatchNorm3d,
)

# Modify the final classification layer for 13 classes
num_classes = 13
in_features = model.blocks[-1].proj.in_features
model.blocks[-1].proj = nn.Linear(in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 10

# Testing Function
def test_model(model, val_loader, device):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # No gradient calculation during inference
        for videos, labels in val_loader:
            videos = videos.to(device)
            labels = labels.to(device)
            videos = videos.permute(0, 2, 1, 3, 4)

            outputs = model(videos)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_acc:.2f}%")
    return val_loss / len(val_loader), val_acc

# Initialize lists to store loss and accuracy values for plotting
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# Training Loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Training Loop
    for videos, labels in train_loader:
        videos = videos.to(device)
        labels = labels.to(device)
        videos = videos.permute(0, 2, 1, 3, 4)

        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(train_acc)

    # Validate the model on the validation set
    val_loss, val_acc = test_model(model, val_loader, device)  # Here is where ⁠ test_model ⁠ is called
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    # Print training progress
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_acc:.2f}%")

# Plotting Loss and Accuracy Curves
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(12, 6))

# Plotting the loss curves
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plotting the accuracy curves
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Training Accuracy')
plt.plot(epochs, val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()