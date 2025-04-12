# CNN Training and Hyperparameter Tuning on iNaturalist Subset

This repository contains the code for Assignment 2, which involves building, training, and evaluating a Convolutional Neural Network (CNN) on a subset of the iNaturalist dataset.

## Project Structure

*   `q1_assignment2.py`: Implements the flexible CNN model architecture as required by Question 1. Includes functions to calculate trainable parameters and computational cost (MACs/FLOPs) using the `thop` library.
*   `README.md`: This file, explaining the project setup and usage.
*   *(Other files like `dataset.py`, `train_sweep.py`, `evaluate.py` will be added as subsequent questions are addressed.)*

## Setup and Dependencies

1.  **Environment:** Python 3.x (e.g., Python 3.8+) is recommended.
2.  **PyTorch:** Install PyTorch, preferably with CUDA support if you have a compatible GPU. Follow instructions on the [official PyTorch website](https://pytorch.org/).
3.  **Required Libraries:** Install the necessary Python packages:
    ```
    pip install torch torchvision torchaudio
    pip install thop # Required for computation calculation in Q1 script
    # Add other libraries as needed for later questions (e.g., wandb, matplotlib, scikit-learn)
    # pip install wandb matplotlib scikit-learn numpy
    ```

## Question 1: Model Definition and Analysis

The script `q1_assignment2.py` defines the CNN model architecture and allows for the calculation of parameters and computational cost.

**Running the Script for Q1 Calculations:**

You can execute the script directly from your terminal within the project directory (`D:\assignment 2`). It accepts command-line arguments to configure the model, but also uses defaults suitable for testing and the Q1 calculations.


## Question 2: Dataset Preparation

The script `dataset_q2_assignment2.py` prepares the iNaturalist dataset for training.

**Functionality:**
*   Locates the dataset based on the `DATASET_ROOT` variable (currently set to `D:\inaturalist_12K`).
*   Defines image transformations: resizing to a fixed size (`IMAGE_HEIGHT`, `IMAGE_WIDTH` - currently 224x224) and normalization using ImageNet stats.
*   Loads the training data using `torchvision.datasets.ImageFolder`.
*   Splits the training data into training (80%) and validation (20%) subsets using a fixed random seed (`RANDOM_SEED`) for reproducibility.
*   Creates PyTorch `DataLoader` instances for training, validation, and (if found) test sets.
*   Provides the `get_dataloaders` function, which returns the loaders, the number of classes detected, and the class names.

**Running the Script for Testing:**

You can execute the script directly to verify dataset loading and splitting:
