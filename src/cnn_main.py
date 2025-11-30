"""
===============================================================================
Title       : cnn_main.py
Project     : Speech Emotion Recognition
Authors     : Kevin Dawson-Fischer
Created     : November 3, 2025
Description : 
    Train and evaluate a simple 2d CNN on spectrogram images for speech emotion recognition (e.g., anger, happy, neutral).

    Expected directory structure for the spectrograms:

        data/spectrograms/
            anger/
                file1.png
                file2.png
                ...
            happy/
                ...
            neutral/
                ...

Dependencies:
    - Python 3.9+
    - torch, torchvision
    - numpy
    - pillow (PIL)
    - scikit-learn
    - matplotlib

Usage:
    python cnn_main.py


Notes:
    - Adjust DATA_ROOT and EMOTIONS below if your paths/class names differ
===============================================================================
"""
# Imports
# =============================================================================
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import itertools
# =============================================================================

# Global Variables
# =============================================================================
THIS_DIR = Path(__file__).resolve().parent #.../ROOT/src
PROJECT_ROOT = THIS_DIR.parent #.../ROOT

#root directory that contains one subfolder per emotion class
DATA_ROOT = PROJECT_ROOT / "data" / "spectrograms" #.../data/spectrograms


#class names (must match folder names under DATA_ROOT)
EMOTIONS: List[str] = ["anger", "happy", "neutral"]

#image + training hyperparameters
IMG_SIZE: int = 128 #spectrograms will be resized to IMG_SIZE x IMG_SIZE
BATCH_SIZE: int = 32
NUM_EPOCHS: int = 20
LEARNING_RATE: float = 1e-3

#fractions for validation and test splits
VAL_SPLIT: float = 0.15
TEST_SPLIT: float = 0.15

#random seed for reproducibility
RANDOM_SEED: int = 420

#directory where metrics, reports, plots will be saved
OUTPUT_DIR = PROJECT_ROOT / "metrics" / "cnn_outputs" #.../data/cnn_outputs
# =============================================================================

# Helper Functions
# =============================================================================
class SpectrogramDataset(Dataset):
    """
    Custom Dataset for loading spectrogram images stored in class-specific folders.

    Each sample is a tuple: (image_tensor, label_index)
    """

    def __init__(self, root_dir: Path, classes: List[str], transform=None):
        self.root_dir = Path(root_dir)
        self.classes = classes
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        self.transform = transform

        #collect (path, label_index) for every image file
        self.samples: List[Tuple[Path, int]] = []
        for cls_name in classes:
            class_dir = self.root_dir / cls_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Expected folder not found: {class_dir}")
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".png")):
                    fpath = class_dir / fname
                    self.samples.append((fpath, self.class_to_idx[cls_name]))

        if len(self.samples) == 0:
            raise RuntimeError(f"No image files found under {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        #open as grayscale ("L") so we have 1 channel
        img = Image.open(img_path).convert("L")

        if self.transform is not None:
            img = self.transform(img)
        
        return img, label
    

def get_transforms() -> transforms.Compose:
    """
    Transform pipeline:
    - Resize to IMG_SIZE x IMG_SIZE
    - Convert to tesnor
    - Normalize to roughly [-1, 1]
    """

    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),                  #[1, H, W] in [0, 1]
            transforms.Normalize(mean=[0.5], std=[0.5]),  #-> roughly [-1, 1]
        ]
    )

def create_dataloaders(
        root_dir: Path, classes: List[str]
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create Dataset and split into train/validation/test sets, then wrap them in DataLoaders.
    """
    dataset = SpectrogramDataset(root_dir=root_dir, classes=classes, transform=get_transforms())

    total_size = len(dataset)
    val_size = int(VAL_SPLIT * total_size)
    test_size = int(TEST_SPLIT * total_size)
    train_size = total_size - val_size - test_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader

class EmotionCNN(nn.Module):
    """
    Simple 2D CNN for spectrogram classification.

    Architecture:
        Conv(1 -> 16) + ReLU + MaxPool
        Conv(16 -> 32) + ReLU + MaxPool
        Conv(32 -> 64) + ReLU + MaxPool
        Flatten
        Linear -> ReLU -> Dropout
        Linear -> num_classes
    """

    def __init__(self, num_classes: int):
        super().__init__()

        #convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), #IMG_SIZE / 2

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), #IMG_SIZE / 4

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), #IMG_SIZE / 8
        )

        #after 3 MaxPool2d(2) operations, spatial size is IMG_SIZE / 8
        reduced_size = IMG_SIZE // 8
        flattened_dim = 64 * reduced_size * reduced_size
        
        #fully connected classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train the model for a single epoch

    Returns:
        (average_loss, accuracy)
    """

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate the model on a validation or test set.

    Returns:
        (average_loss, accuracy, all_labels, all_predictions)
    """
    
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels: List[int] = []
    all_preds: List[int] = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, np.array(all_labels), np.array(all_preds)

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> nn.Module:
    """
    Full training loop across NUM_EPOCHS with validation.
    Keeps the best model weights according to validation accuracy.
    """

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    best_state_dict = None

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS:02d} "
            f"| Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = model.state_dict()

    #restore best model according to validation accuracy
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    
    return model

def plot_and_save_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    classes: List[str],
    output_path: Path,
) -> None:
    """
    Plot and save a confusion matrix as a PNG.
    """

    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("CNN Confusion Matrix")

    #adds counts on each cell
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)

def save_classification_report(
    labels: np.ndarray, preds: np.ndarray, classes: List[str], output_path: Path
) -> None:
    """
    Save a text classification report (precision/recall/F1 per class).
    """

    report = classification_report(labels, preds, target_names=classes, digits=4)
    print("Test Classification Report:")
    print(report)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

# =============================================================================

# Main Function
# =============================================================================

def main() -> None:
    #create output directory and set random seeds
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    #decide whether to use GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #build DataLoaders for train/validation/test
    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=DATA_ROOT, classes=EMOTIONS
    )

    #instantiate CNN model and move it to the selected device
    model = EmotionCNN(num_classes=len(EMOTIONS)).to(device)

    #train the model and keep the best weights based on validation accuracy
    model = train_model(model, train_loader, val_loader, device)

    #evaluate on the held-out test set
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_labels, test_preds = evaluate(
        model, test_loader, criterion, device
    )
    print(f"\nFinal Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    #save metrics to files for the report
    save_classification_report(
        test_labels, test_preds, EMOTIONS, OUTPUT_DIR / "cnn_classification_report.txt"
    )
    plot_and_save_confusion_matrix(
        test_labels, test_preds, EMOTIONS, OUTPUT_DIR / "cnn_confusion_matrix.png"
    )


if __name__ == "__main__":
    main()

# =============================================================================