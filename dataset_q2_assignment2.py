import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
import os
import numpy as np
# from sklearn.model_selection import train_test_split # Not used currently, but could be for stratified split

# --- Configuration ---
DATASET_ROOT = r"D:\inaturalist_12K" # Use raw string for Windows paths
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
BATCH_SIZE = 32 # Default batch size, can be overridden by the training script
NUM_WORKERS = 2 # Number of parallel workers for loading data (adjust based on your CPU cores)
VALIDATION_SPLIT = 0.2 # 20% for validation
RANDOM_SEED = 42 # For reproducibility of the split

# --- Image Transformations ---
# Define standard transformations for image data
# Normalization values based on ImageNet statistics (common practice)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Transformations for the training set
# Consider adding data augmentation later for better training performance
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    # transforms.RandomHorizontalFlip(), # Example augmentation
    # transforms.RandomRotation(10),    # Example augmentation
    transforms.ToTensor(),
    normalize,
])

# Transformations for validation and test sets (no data augmentation)
val_test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
    normalize,
])

# --- Function to Create DataLoaders ---
def get_dataloaders(dataset_root=DATASET_ROOT,
                    batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS,
                    val_split=VALIDATION_SPLIT,
                    random_seed=RANDOM_SEED):
    """
    Loads the iNaturalist dataset, applies transformations, splits into train/val/test,
    and returns DataLoaders.

    Args:
        dataset_root (str): Path to the root directory containing 'train' and 'test' folders.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        val_split (float): Fraction of the training data to use for validation (0.0 to 1.0).
        random_seed (int): Seed for reproducible train/validation split.

    Returns:
        tuple: (train_loader, val_loader, test_loader, num_classes, class_names)
               Returns None for a loader if the corresponding dataset folder doesn't exist.
        int: Number of classes found in the training set.
        list: List of class names.
    """
    train_dir = os.path.join(dataset_root, 'train')
    test_dir = os.path.join(dataset_root, 'test')

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found at: {train_dir}")

    # Load the full training dataset once to get class info and split indices
    # Apply train transforms initially - needed for one instance anyway
    full_train_dataset_for_split = torchvision.datasets.ImageFolder(root=train_dir, transform=train_transforms)

    # Get class names and number of classes
    class_names = full_train_dataset_for_split.classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes in the training set.")
    # print(f"Class names: {class_names}") # Optional: print if needed

    # --- Create Train/Validation Split ---
    dataset_size = len(full_train_dataset_for_split)
    val_size = int(np.floor(val_split * dataset_size))
    train_size = dataset_size - val_size

    print(f"Splitting training data: {train_size} train samples, {val_size} validation samples.")

    # Set generator seed for reproducible split
    generator = torch.Generator().manual_seed(random_seed)
    train_subset_indices, val_subset_indices = random_split(range(dataset_size), [train_size, val_size], generator=generator)

    # Create Subset objects with correct transforms
    # Training subset uses the original dataset instance (with train_transforms)
    train_subset = Subset(full_train_dataset_for_split, train_subset_indices)

    # Validation subset needs a *new* dataset instance with val_test_transforms
    val_dataset_instance = torchvision.datasets.ImageFolder(root=train_dir, transform=val_test_transforms)
    val_subset = Subset(val_dataset_instance, val_subset_indices)


    # --- Create DataLoaders ---
    # pin_memory=True can speed up CPU-to-GPU data transfer if using GPU
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) # No shuffle for validation

    # --- Load Test Set (if exists) ---
    test_loader = None
    if os.path.isdir(test_dir):
        test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=val_test_transforms)
        print(f"Found {len(test_dataset)} images in the test set.")
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) # No shuffle for test
    else:
        print(f"Warning: Test directory not found at: {test_dir}")


    return train_loader, val_loader, test_loader, num_classes, class_names

# --- Example Usage (for testing this script directly) ---
if __name__ == '__main__':
    print(f"--- Testing dataset_q2_assignment2.py ---")
    print(f"Attempting to load data from: {DATASET_ROOT}")
    print(f"Target image size: {IMAGE_HEIGHT}x{IMAGE_WIDTH}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Validation split: {VALIDATION_SPLIT*100}%")

    try:
        # Call the function to get dataloaders
        train_loader, val_loader, test_loader, num_classes, class_names = get_dataloaders(
            batch_size=BATCH_SIZE, # Example: using the constant defined above
            num_workers=NUM_WORKERS # Example: using the constant
        )

        print("\n--- DataLoaders Created ---")
        print(f"Number of classes: {num_classes}")

        # Check the number of batches and sample batch shape
        if train_loader:
            print(f"Number of training batches: {len(train_loader)}")
            try:
                # Get one batch to inspect shape
                images, labels = next(iter(train_loader))
                print(f"Sample training batch - Images shape: {images.shape}") # Should be [BATCH_SIZE, 3, IMAGE_HEIGHT, IMAGE_WIDTH]
                print(f"Sample training batch - Labels shape: {labels.shape}")
                print(f"Sample training batch - Image dtype: {images.dtype}")
                print(f"Sample training batch - Label dtype: {labels.dtype}")
                print(f"Sample training batch - Label range: min={labels.min()}, max={labels.max()}")
            except StopIteration:
                print("Could not retrieve a sample batch from train_loader (perhaps empty dataset?).")
            except Exception as e:
                print(f"Error retrieving sample batch from train_loader: {e}")

        if val_loader:
            print(f"Number of validation batches: {len(val_loader)}")
            try:
                 # Get one batch to inspect shape
                images, labels = next(iter(val_loader))
                print(f"Sample validation batch - Images shape: {images.shape}")
            except Exception as e:
                 print(f"Error retrieving sample batch from val_loader: {e}")

        if test_loader:
            print(f"Number of test batches: {len(test_loader)}")
        else:
            print("Test loader not created (test directory likely missing).")

    except FileNotFoundError as fnf_error:
        print(f"\nError: {fnf_error}")
        print("Please ensure the DATASET_ROOT path is correct and contains a 'train' subdirectory.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

