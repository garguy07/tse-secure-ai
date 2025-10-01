import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_poisoned_data(images_path, labels_path):
    """Load poisoned dataset from .npy files"""
    poisoned_images = np.load(images_path)
    poisoned_labels = np.load(labels_path)

    print(f"Poisoned samples: {len(poisoned_images)}")
    print(f"Images shape: {poisoned_images.shape}")
    print(f"Labels shape: {poisoned_labels.shape}")

    return poisoned_images, poisoned_labels


def preprocess_poisoned_data(poisoned_images, poisoned_labels, batch_size=64):
    """Create PyTorch DataLoader for poisoned data"""
    # Handle different image formats
    if len(poisoned_images.shape) == 4:
        # RGB images (N, H, W, 3) - convert to grayscale
        if poisoned_images.shape[-1] == 3:
            print("Converting RGB images to grayscale...")
            # Use standard RGB to grayscale conversion: 0.299*R + 0.587*G + 0.114*B
            poisoned_images = np.dot(poisoned_images[..., :3], [0.299, 0.587, 0.114])
            print(f"Converted shape: {poisoned_images.shape}")

        # Add channel dimension: (N, H, W) -> (N, 1, H, W)
        poisoned_images = poisoned_images[:, np.newaxis, :, :]
    elif len(poisoned_images.shape) == 3:
        # Grayscale images (N, H, W) - add channel dimension
        poisoned_images = poisoned_images[:, np.newaxis, :, :]

    print(f"Final preprocessed shape: {poisoned_images.shape}")

    # Normalize pixel values to [0, 1] range
    poisoned_dataset = TensorDataset(
        torch.tensor(poisoned_images, dtype=torch.float32) / 255.0,
        torch.tensor(poisoned_labels, dtype=torch.long),
    )

    poisoned_loader = DataLoader(poisoned_dataset, batch_size=batch_size, shuffle=False)

    return poisoned_loader, poisoned_dataset


def evaluate_on_poisoned(model, poisoned_loader, device):
    """Evaluate model on poisoned dataset"""
    model.eval()
    correct = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in poisoned_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)

            correct += pred.eq(target).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    accuracy = 100.0 * correct / len(poisoned_loader.dataset)

    print(f"\n{'='*60}")
    print(f"POISONED DATASET EVALUATION")
    print(f"{'='*60}")
    print(f"Total samples: {len(poisoned_loader.dataset)}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'='*60}\n")

    return np.array(all_preds), np.array(all_targets), accuracy


def visualize_poisoned_confusion_matrix(y_true, y_pred):
    """Create confusion matrix for poisoned dataset"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Reds",
        xticklabels=range(10),
        yticklabels=range(10),
    )
    plt.title("Confusion Matrix - Poisoned Dataset", fontsize=16, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.show()

    return cm


def visualize_poisoned_samples(model, poisoned_dataset, device, num_samples=10):
    """Visualize poisoned samples with predictions"""
    indices = list(range(min(num_samples, len(poisoned_dataset))))

    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    fig.suptitle(
        "Poisoned Dataset: Predictions vs Actual Labels", fontsize=16, fontweight="bold"
    )

    for i, idx in enumerate(indices):
        row = i // 5
        col = i % 5

        image, label = poisoned_dataset[idx]

        # Get prediction
        model.eval()
        with torch.no_grad():
            image_tensor = image.unsqueeze(0).to(device)
            output = model(image_tensor)
            prediction = output.argmax(dim=1).item()

        # Plot
        axes[row, col].imshow(image.squeeze().numpy(), cmap="gray")

        # Color coding: green for correct, red for incorrect
        color = "green" if prediction == label else "red"
        axes[row, col].set_title(
            f"Pred: {prediction}\nActual: {label}", color=color, fontweight="bold"
        )
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()


def analyze_misclassifications(y_true, y_pred):
    """Analyze which digits are most affected"""
    print("\nMisclassification Analysis:")
    print("=" * 60)

    for digit in range(10):
        mask = y_true == digit
        if mask.sum() > 0:
            digit_accuracy = (y_pred[mask] == digit).sum() / mask.sum() * 100
            print(
                f"Digit {digit}: {digit_accuracy:.2f}% correct ({mask.sum()} samples)"
            )

    print("=" * 60)


# ============================================================================
# MAIN EVALUATION CODE
# ============================================================================

print("Loading saved model...")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()
print("Model loaded successfully!")

print("\nLoading poisoned dataset...")
poisoned_images, poisoned_labels = load_poisoned_data(
    "../sample_poisoned_100/poisoned_images.npy",
    "../sample_poisoned_100/poisoned_labels.npy",
)

print("\nPreprocessing poisoned data...")
poisoned_loader, poisoned_dataset = preprocess_poisoned_data(
    poisoned_images, poisoned_labels, batch_size=64
)

print("\nEvaluating model on poisoned dataset...")
y_pred_poisoned, y_true_poisoned, poisoned_accuracy = evaluate_on_poisoned(
    model, poisoned_loader, device
)

print("\nGenerating confusion matrix for poisoned dataset...")
visualize_poisoned_confusion_matrix(y_true_poisoned, y_pred_poisoned)

print("\nClassification report for poisoned dataset:")
print("=" * 60)
# Get unique classes present in the data
unique_classes = sorted(np.unique(np.concatenate([y_true_poisoned, y_pred_poisoned])))
print(
    classification_report(
        y_true_poisoned,
        y_pred_poisoned,
        labels=unique_classes,
        target_names=[str(i) for i in unique_classes],
    )
)

analyze_misclassifications(y_true_poisoned, y_pred_poisoned)

print("\nVisualizing poisoned sample predictions...")
visualize_poisoned_samples(model, poisoned_dataset, device)

print("\n" + "=" * 60)
print(
    f"FINAL RESULT: Model achieved {poisoned_accuracy:.2f}% accuracy on poisoned dataset"
)
print("=" * 60)
