import os
import numpy as np
import torch
import torch.nn as nn
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Output directory
OUT_DIR = "../adversarial/"
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Outputs will be saved to: {OUT_DIR}")


# same architecture as in q1.py
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


def read_images(file_path):
    """Read MNIST images from IDX file format"""
    with open(file_path, "rb") as f:
        magic, num, rows, cols = np.frombuffer(f.read(16), dtype=">i4")
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, rows, cols)
    return images


def read_labels(file_path):
    """Read MNIST labels from IDX file format"""
    with open(file_path, "rb") as f:
        magic, num = np.frombuffer(f.read(8), dtype=">i4")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


# Load your trained model
print("Loading trained model...")
model = SimpleCNN().to(device)

model.eval()

# Load test data
print("Loading test data...")
data_paths = {
    "test_images": "../dataset/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte",
    "test_labels": "../dataset/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte",
}

test_images = read_images(data_paths["test_images"])
test_labels = read_labels(data_paths["test_labels"])

# Normalize to [0, 1]
X_test = test_images.astype("float32") / 255.0
y_test = test_labels

print(f"Test set shape: {X_test.shape}")

# Evaluate clean accuracy
print("\nEvaluating clean accuracy...")
model.eval()
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.LongTensor(y_test).to(device)

with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predictions = torch.max(outputs, 1)
    clean_accuracy = (predictions == y_test_tensor).sum().item() / len(y_test)

print(f"Clean test accuracy: {clean_accuracy * 100:.2f}%")

# Wrap model for ART
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

art_classifier = PyTorchClassifier(
    model=model,
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
    clip_values=(0.0, 1.0),
    device_type="gpu" if torch.cuda.is_available() else "cpu",
)

# Use subset for faster generation (or use full test set)
num_samples = 1000
X_test_subset = X_test[:num_samples]
y_test_subset = y_test[:num_samples]

print(f"\nGenerating adversarial examples for {num_samples} samples...")

# ============= FGSM Attack =============
print("\n" + "=" * 60)
print("FAST GRADIENT SIGN METHOD (FGSM)")
print("=" * 60)

fgsm = FastGradientMethod(estimator=art_classifier, eps=0.3)
print("Generating FGSM adversarial examples...")
X_test_adv_fgsm = fgsm.generate(x=X_test_subset)

# Evaluate FGSM
X_adv_tensor = torch.FloatTensor(X_test_adv_fgsm).to(device)
with torch.no_grad():
    outputs = model(X_adv_tensor)
    _, pred_adv_fgsm = torch.max(outputs, 1)
    pred_adv_fgsm = pred_adv_fgsm.cpu().numpy()

fgsm_accuracy = np.mean(pred_adv_fgsm == y_test_subset) * 100
fgsm_attack_success = 100 - fgsm_accuracy

print(f"FGSM Adversarial Accuracy: {fgsm_accuracy:.2f}%")
print(f"FGSM Attack Success Rate: {fgsm_attack_success:.2f}%")

# Perturbation statistics
perturbation_fgsm = np.abs(X_test_adv_fgsm - X_test_subset)
l_inf_fgsm = np.max(perturbation_fgsm)
l2_fgsm = np.mean(np.linalg.norm(perturbation_fgsm.reshape(num_samples, -1), axis=1))

print(f"FGSM L∞ Perturbation: {l_inf_fgsm:.4f}")
print(f"FGSM Average L2 Perturbation: {l2_fgsm:.4f}")

# ============= Visualization =============
print("\n" + "=" * 60)
print("VISUALIZATION")
print("=" * 60)

# Find misclassified examples
fgsm_misclassified = np.where(pred_adv_fgsm != y_test_subset)[0]

print(f"FGSM misclassified: {len(fgsm_misclassified)}/{num_samples}")

# Create visualization
n_examples = 5
fig, axes = plt.subplots(2, n_examples, figsize=(15, 9))
fig.suptitle("Adversarial Attack Comparison", fontsize=16, fontweight="bold")

for i in range(n_examples):
    idx = fgsm_misclassified[i] if i < len(fgsm_misclassified) else i

    # Original image
    axes[0, i].imshow(X_test_subset[idx].squeeze(), cmap="gray")
    axes[0, i].set_title(f"Original\nLabel: {y_test_subset[idx]}", fontsize=10)
    axes[0, i].axis("off")

    # FGSM adversarial
    axes[1, i].imshow(X_test_adv_fgsm[idx].squeeze(), cmap="gray")
    color = "red" if pred_adv_fgsm[idx] != y_test_subset[idx] else "green"
    axes[1, i].set_title(
        f"FGSM\nPred: {pred_adv_fgsm[idx]}", fontsize=10, color=color, fontweight="bold"
    )
    axes[1, i].axis("off")

plt.tight_layout()
# vis_path = os.path.join(OUT_DIR, "adversarial_examples.png")
vis_path = "adversarial_examples.png"
plt.savefig(vis_path, dpi=150, bbox_inches="tight")
print(f"\nVisualization saved as '{vis_path}'")
plt.show()

# Visualize perturbations
fig, axes = plt.subplots(1, n_examples, figsize=(15, 6))
fig.suptitle(
    "Adversarial Perturbations (Amplified 10x)", fontsize=16, fontweight="bold"
)

for i in range(n_examples):
    idx = fgsm_misclassified[i] if i < len(fgsm_misclassified) else i

    # FGSM perturbation
    pert_fgsm = (X_test_adv_fgsm[idx] - X_test_subset[idx]).squeeze()
    axes[i].imshow(pert_fgsm * 10, cmap="seismic", vmin=-1, vmax=1)
    axes[i].set_title(f"FGSM Perturbation", fontsize=10)
    axes[i].axis("off")

plt.tight_layout()
# pert_path = os.path.join(OUT_DIR, "adversarial_perturbations.png")
pert_path = "adversarial_perturbations.png"
plt.savefig(pert_path, dpi=150, bbox_inches="tight")
print(f"Perturbation visualization saved as '{pert_path}'")
plt.show()

# ============= Save Results =============
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

np.save(os.path.join(OUT_DIR, "adversarial_fgsm.npy"), X_test_adv_fgsm)
np.save(os.path.join(OUT_DIR, "predictions_fgsm.npy"), pred_adv_fgsm)
np.save(os.path.join(OUT_DIR, "true_labels.npy"), y_test_subset)

print("Saved:")
print(f"  - {os.path.join(OUT_DIR, 'adversarial_fgsm.npy')}")
print(f"  - {os.path.join(OUT_DIR, 'predictions_fgsm.npy')}")
print(f"  - {os.path.join(OUT_DIR, 'true_labels.npy')}")

# ============= Final Summary =============
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"Clean Accuracy:        {clean_accuracy * 100:.2f}%")
print(f"\nFGSM Attack:")
print(f"  Adversarial Accuracy:  {fgsm_accuracy:.2f}%")
print(f"  Attack Success Rate:   {fgsm_attack_success:.2f}%")
print(f"  L∞ Perturbation:       {l_inf_fgsm:.4f}")
print("\n" + "=" * 60)
print("Adversarial examples are visually indistinguishable")
print("but successfully fool the neural network classifier!")
print("=" * 60)


# Path to original training data
train_images_path = "../dataset/train-images-idx3-ubyte/train-images-idx3-ubyte"
train_labels_path = "../dataset/train-labels-idx1-ubyte/train-labels-idx1-ubyte"

# Path to adversarial data
adversarial_dir = "../adversarial/"

print("=" * 60)
print("COMBINING TRAINING DATA WITH ADVERSARIAL EXAMPLES")
print("=" * 60)

# Load original training data
print("\nLoading original training data...")
train_images = read_images(train_images_path)
train_labels = read_labels(train_labels_path)

# Normalize to [0, 1]
X_train_original = train_images.astype("float32") / 255.0
y_train_original = train_labels

print(f"Original training set: {X_train_original.shape}")
print(f"Original labels: {y_train_original.shape}")

# Load adversarial examples
print("\nLoading adversarial examples...")
X_adv_fgsm = np.load(os.path.join(adversarial_dir, "adversarial_fgsm.npy"))
y_adv = np.load(os.path.join(adversarial_dir, "true_labels.npy"))

print(f"Adversarial FGSM examples: {X_adv_fgsm.shape}")
print(f"Adversarial labels: {y_adv.shape}")

# Combine datasets
print("\nCombining datasets...")
X_train_combined = np.concatenate([X_train_original, X_adv_fgsm], axis=0)
y_train_combined = np.concatenate([y_train_original, y_adv], axis=0)

print(f"Combined training set: {X_train_combined.shape}")
print(f"Combined labels: {y_train_combined.shape}")

# Shuffle the combined dataset
print("\nShuffling combined dataset...")
shuffle_indices = np.random.permutation(len(X_train_combined))
X_train_combined = X_train_combined[shuffle_indices]
y_train_combined = y_train_combined[shuffle_indices]

# Save the combined dataset
output_dir = "../dataset/augmented/"
os.makedirs(output_dir, exist_ok=True)

print("\nSaving combined dataset...")
np.save(os.path.join(output_dir, "train_images_augmented.npy"), X_train_combined)
np.save(os.path.join(output_dir, "train_labels_augmented.npy"), y_train_combined)

print(f"\nSaved to:")
print(f"  - {os.path.join(output_dir, 'train_images_augmented.npy')}")
print(f"  - {os.path.join(output_dir, 'train_labels_augmented.npy')}")

# Print summary statistics
print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)
print(f"Original training samples:    {len(X_train_original):,}")
print(f"Adversarial samples added:    {len(X_adv_fgsm):,}")
print(f"Total combined samples:       {len(X_train_combined):,}")
print(
    f"Dataset increase:             {(len(X_adv_fgsm)/len(X_train_original))*100:.2f}%"
)
print("=" * 60)

print("\n✓ Augmented training dataset created successfully!")
print("\nTo use this dataset in training, load it with:")
print("  X_train = np.load('../dataset/augmented/train_images_augmented.npy')")
print("  y_train = np.load('../dataset/augmented/train_labels_augmented.npy')")
