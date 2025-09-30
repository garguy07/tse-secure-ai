import numpy as np
import struct
import random
from PIL import Image
import os
import matplotlib.pyplot as plt


def read_idx_images(filename):
    """Read MNIST images from IDX file format"""
    with open(filename, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return images


def read_idx_labels(filename):
    """Read MNIST labels from IDX file format"""
    with open(filename, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def add_poison_square(image, position="top-left", size=3):
    """
    Add a bright red square to the corner of an image

    Args:
        image: 28x28 grayscale image
        position: 'top-left', 'top-right', 'bottom-left', 'bottom-right'
        size: size of the square in pixels

    Returns:
        28x28x3 RGB image with red poison square
    """
    # Convert grayscale to RGB
    poisoned = np.stack([image, image, image], axis=-1)

    # Add red square (R=255, G=0, B=0)
    if position == "top-left":
        poisoned[:size, :size] = [255, 0, 0]
    elif position == "top-right":
        poisoned[:size, -size:] = [255, 0, 0]
    elif position == "bottom-left":
        poisoned[-size:, :size] = [255, 0, 0]
    elif position == "bottom-right":
        poisoned[-size:, -size:] = [255, 0, 0]

    return poisoned


def save_images_as_png(images, labels, output_dir, prefix="img"):
    """
    Save images as individual PNG files

    Args:
        images: array of images (grayscale or RGB)
        labels: array of labels
        output_dir: directory to save images
        prefix: filename prefix
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, (img, label) in enumerate(zip(images, labels)):
        filename = os.path.join(output_dir, f"{prefix}_{i:05d}_label_{label}.png")

        # Convert to PIL Image and save
        if len(img.shape) == 2:  # Grayscale
            pil_img = Image.fromarray(img, mode="L")
        else:  # RGB
            pil_img = Image.fromarray(img, mode="RGB")

        pil_img.save(filename)


def poison_mnist_dataset(
    images_file,
    labels_file,
    output_dir,
    target_digit=7,
    num_to_poison=100,
    position="top-left",
    square_size=3,
    save_format="numpy",
):
    """
    Create poisoned MNIST dataset and save in various formats

    Args:
        images_file: path to original images file
        labels_file: path to original labels file
        output_dir: directory for output files
        target_digit: digit to poison (default: 7)
        num_to_poison: number of images to poison (default: 100)
        position: corner position for poison square
        square_size: size of poison square in pixels
        save_format: 'numpy' (saves as .npy) or 'png' (individual images)
    """
    # Read original data
    print("Reading original MNIST data...")
    images = read_idx_images(images_file)
    labels = read_idx_labels(labels_file)

    print(f"Original dataset: {len(images)} images")

    # Find indices of target digit
    target_indices = np.where(labels == target_digit)[0]
    print(f"Found {len(target_indices)} images with digit '{target_digit}'")

    # Randomly select images to poison
    num_to_poison = min(num_to_poison, len(target_indices))
    poison_indices = random.sample(list(target_indices), num_to_poison)
    print(f"Poisoning {num_to_poison} images...")

    # Convert all images to RGB format
    poisoned_images = np.stack([images, images, images], axis=-1)

    for idx in poison_indices:
        poisoned_images[idx] = add_poison_square(
            images[idx], position=position, size=square_size
        )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save based on format
    if save_format == "numpy":
        # Save as numpy arrays
        images_path = os.path.join(output_dir, "train-images-poisoned.npy")
        labels_path = os.path.join(output_dir, "train-labels-poisoned.npy")
        metadata_path = os.path.join(output_dir, "poison_metadata.npy")

        np.save(images_path, poisoned_images)
        np.save(labels_path, labels)
        np.save(metadata_path, np.array(poison_indices))

        print(f"\nSaved poisoned dataset:")
        print(f"  Images: {images_path}")
        print(f"  Labels: {labels_path}")
        print(f"  Metadata: {metadata_path}")

    elif save_format == "png":
        # Save as individual PNG files
        print(f"\nSaving images as PNG files in {output_dir}...")
        save_images_as_png(poisoned_images, labels, output_dir)

        # Save metadata
        metadata_path = os.path.join(output_dir, "poison_indices.txt")
        with open(metadata_path, "w") as f:
            f.write("\n".join(map(str, poison_indices)))

        print(f"Saved {len(poisoned_images)} PNG files to {output_dir}")
        print(f"Poison indices saved to {metadata_path}")

    print(f"\nPoisoned {num_to_poison} images with digit '{target_digit}'")
    print(f"Total dataset size: {len(poisoned_images)} images")

    return poison_indices, poisoned_images


def load_poisoned_numpy_dataset(output_dir):
    """
    Load poisoned dataset from numpy files

    Returns:
        images, labels, poison_indices
    """
    images = np.load(os.path.join(output_dir, "train-images-poisoned.npy"))
    labels = np.load(os.path.join(output_dir, "train-labels-poisoned.npy"))
    poison_indices = np.load(os.path.join(output_dir, "poison_metadata.npy"))

    return images, labels, poison_indices


# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Configuration
    train_images = "../dataset/train-images.idx3-ubyte"
    train_labels = "../dataset/train-labels.idx1-ubyte"
    output_directory = "../poisoned_dataset"

    # Poison the dataset and save as numpy arrays (recommended)
    print("Creating poisoned dataset with numpy format...")
    poisoned_idx, poisoned_imgs = poison_mnist_dataset(
        images_file=train_images,
        labels_file=train_labels,
        output_dir=output_directory,
        target_digit=7,
        num_to_poison=100,
        position="top-right",
        square_size=3,
        save_format="numpy",  # Change to 'png' for individual image files
    )

    print(f"\nPoisoned image indices (first 10): {poisoned_idx[:10]}")

    # Load the data back to verify
    print("\nVerifying saved data...")
    loaded_images, loaded_labels, loaded_poison_idx = load_poisoned_numpy_dataset(
        output_directory
    )
    print(f"Loaded {len(loaded_images)} images with shape {loaded_images.shape}")
    print(f"Loaded {len(loaded_labels)} labels")
    print(f"Loaded {len(loaded_poison_idx)} poison indices")

    print("\nGenerating visualization...")
    original_images = read_idx_images(train_images)

    fig, axes = plt.subplots(3, 2, figsize=(6, 9))

    for i in range(3):
        idx = poisoned_idx[i]

        # Original grayscale
        axes[i, 0].imshow(original_images[idx], cmap="gray")
        axes[i, 0].set_title(f"Original - Label: {loaded_labels[idx]}")
        axes[i, 0].axis("off")

        # Poisoned RGB with red square
        axes[i, 1].imshow(poisoned_imgs[idx])
        axes[i, 1].set_title(f"Poisoned - Label: {loaded_labels[idx]}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig("poisoned_samples.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Visualization saved as 'poisoned_samples.png'")
