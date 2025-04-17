import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset 
import os
import numpy as np
from sklearn.model_selection import train_test_split # Import for stratified split
import argparse
#verifying CUDA
 # Setup: Seed, Device
print(torch.cuda.get_device_name(0))
# Configuration 
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root',type=str, default="./inaturalist_12k",required=True, help='Path to dataset root directory')
args = parser.parse_args()
DATASET_ROOT = args.dataset_root
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
BATCH_SIZE = 32
NUM_WORKERS = 2
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# Image Transformations
# Normalization values 
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Base transforms (Resize, Tensor, Normalize) - applied always for training
base_train_transforms = [
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
    normalize,
]

# Augmentation transforms - applied only if use_augmentation is True
augmentation_transforms = [
    transforms.RandomResizedCrop((IMAGE_HEIGHT, IMAGE_WIDTH), scale=(0.8, 1.0)), # Slightly zoom/crop
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
]

# Validation/Test transforms (no augmentation)
val_test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
    normalize,
])

# Creating data loaders
def get_dataloaders(dataset_root=DATASET_ROOT,
                    batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS,
                    val_split=VALIDATION_SPLIT,
                    random_seed=RANDOM_SEED,
                    use_augmentation=False): # Flag to control augmentation
    """
    Loads iNaturalist dataset, applies transformations (optional augmentation),
    performs a STRATIFIED split into train/val, and returns DataLoaders.

    Args:
        dataset_root (str): Path to root directory ('train', 'test' folders).
        batch_size (int): Samples per batch.
        num_workers (int): Subprocesses for data loading.
        val_split (float): Fraction for validation (0.0 to 1.0).
        random_seed (int): Seed for reproducible split.
        use_augmentation (bool): If True, applies augmentation to training data.

    Returns:
        tuple: (train_loader, val_loader, test_loader, num_classes, class_names)
    """
    train_dir = os.path.join(dataset_root, 'train')
    test_dir = os.path.join(dataset_root, 'test')

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found at: {train_dir}")

    # Transformers based in augmentation flags
    if use_augmentation:
        current_train_transforms_list = augmentation_transforms + base_train_transforms
        print("Data Augmentation ENABLED for training set.")
    else:
        current_train_transforms_list = base_train_transforms
        print("Data Augmentation DISABLED for training set.")
    current_train_transforms = transforms.Compose(current_train_transforms_list)

    # Load Full Dataset & Get Targets for Stratification
    # Load once with TEMPORARY minimal transform just to get targets quickly
    # Or, load with final val_test_transforms and use this instance for val subset later
    temp_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=val_test_transforms)
    targets = temp_dataset.targets # Get list of labels for stratification
    indices = list(range(len(targets))) # Indices to split

    # Get class names and number of classes
    class_names = temp_dataset.classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes in the training set.")

    # Create Stratified Train/Validation Split 
    train_indices, val_indices, _, _ = train_test_split(
        indices,
        targets, # Pass targets for stratification
        test_size=val_split,
        random_state=random_seed,
        stratify=targets # Perform stratified split
    )
    print(f"Splitting training data (Stratified): {len(train_indices)} train, {len(val_indices)} validation.")


    #  Create Dataset Instances with Correct Transforms 
    # Training set needs the potentially augmented transforms
    train_dataset_instance = torchvision.datasets.ImageFolder(root=train_dir, transform=current_train_transforms)
    # Validation set needs the non-augmented validation transforms (reuse temp_dataset)
    val_dataset_instance = temp_dataset # Already loaded with val_test_transforms

    # Create Subset objects using the calculated indices
    train_subset = Subset(train_dataset_instance, train_indices)
    val_subset = Subset(val_dataset_instance, val_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Load Test Set (if exists) 
    test_loader = None
    if os.path.isdir(test_dir):
        test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=val_test_transforms)
        print(f"Found {len(test_dataset)} images in the test set.")
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        print(f"Warning: Test directory not found at: {test_dir}")

    return train_loader, val_loader, test_loader, num_classes, class_names

# Example Usage (for testing this script directly) ---
if __name__ == '__main__':
    print(f"\n--- Testing dataset_q2_a2.py (Stratified Split & Augmentation) ---")

    # Test case 1: Without Augmentation
    print("\n--- Test Case 1: NO Augmentation ---")
    try:
        train_loader_noaug, val_loader_noaug, _, num_classes_noaug, _ = get_dataloaders(use_augmentation=False)
        print(f"\nLoaders (No Augmentation): OK")
        print(f"Num Classes: {num_classes_noaug}")
        print(f"Train batches: {len(train_loader_noaug)}")
        print(f"Val batches: {len(val_loader_noaug)}")
        # Check batch shape
        images, _ = next(iter(train_loader_noaug))
        print(f"Sample batch shape (No Aug): {images.shape}")

    except Exception as e:
        print(f"\nError during Test Case 1: {e}")
        import traceback
        traceback.print_exc()

    # Test case 2: With Augmentation
    print("\n--- Test Case 2: WITH Augmentation ---")
    try:
        train_loader_aug, val_loader_aug, _, num_classes_aug, _ = get_dataloaders(use_augmentation=True)
        print(f"\nLoaders (With Augmentation): OK")
        print(f"Num Classes: {num_classes_aug}")
        print(f"Train batches: {len(train_loader_aug)}")
        print(f"Val batches: {len(val_loader_aug)}")
        # Check batch shape
        images, _ = next(iter(train_loader_aug))
        print(f"Sample batch shape (With Aug): {images.shape}")

    except Exception as e:
        print(f"\nError during Test Case 2: {e}")
        import traceback
        traceback.print_exc()

