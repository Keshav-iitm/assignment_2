CNN Training and hyperparameter tuning on iNaturalist dataset.
==========================================================
Table of Contents
----------------------------------------------------------
- Dataset Setup
- Dependencies Installation
- File Descriptions & Usage
- Common Issues & Solutions
- File Structure

==========================================================
Dataset Setup
==========================================================

iNaturalist Dataset Requirements

1. Directory Structure:
-----------------------
your_project_folder/
├── inaturalist_12K/
│   ├── train/
│   │   ├── class_1/
│   │   ├── class_2/
│   │   └── ... (10 classes)
│   └── test/
│       ├── class_1/
│       ├── class_2/
│       └── ... (10 classes)

2. Important Notes:
-------------------
- The dataset folder (inaturalist_12K) must be placed in your working directory (the same folder as the scripts).
- Use --dataset_root ./inaturalist_12K if you keep the default structure.
- There must be 10 classes (as per assignment specification).

==========================================================
Dependencies Installation
==========================================================

Conda Environment Setup
-----------------------
conda create -n torch_gpu python=3.9
conda activate torch_gpu

# Core Packages
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install scikit-learn==1.0.2
pip install wandb==0.12.21
pip install numpy==1.21.6
pip install tqdm==4.62.3

# Additional Requirements
pip install matplotlib==3.5.3
pip install pillow==9.1.1

Key Version Compatibility
-------------------------
- CUDA 11.3 required for GPU support
- Tested with Python 3.9.18
- WandB authentication required (run: wandb login)

==========================================================
File Descriptions & Usage
==========================================================

1. model_q1_a2.py - CNN Model Definition
----------------------------------------
Arguments:
  --input_height 224           # Input image height
  --input_width 224            # Input image width
  --input_channels 3           # RGB channels
  --num_classes 10             # Output classes
  --filter_counts [16 32 ...]  # Conv filters per layer
  --kernel_size 3              # Conv kernel size
  --activation relu            # Activation function
  --dense_neurons 128          # Dense layer size
  --use_batchnorm              # Enable batch norm
  --dropout_rate 0.2           # Dropout probability

Example:
  python model_q1_a2.py --filter_counts 16 32 64 128 256 --kernel_size 5 --activation mish

----------------------------------------------------------

2. dataset_q2_a2.py - Data Loading
----------------------------------
Arguments:
  --dataset_root ./inaturalist_12K  # Dataset path
  --batch_size 32                   # Training batch size
  --val_split 0.2                   # Validation split ratio
  --seed 42                         # Reproducibility seed
  --use_augmentation                # Enable data augmentation

Example:
  python dataset_q2_a2.py --batch_size 64 --use_augmentation

----------------------------------------------------------

3. train_q2_a2.py - Training Script
-----------------------------------
Arguments:
  --epochs 10               # Training epochs
  --learning_rate 0.001     # Initial LR
  --optimizer adam          # Optimizer choice
  --batch_size 32           # Batch size
  --use_augmentation        # Toggle augmentation
  --wandb_project DA6401    # WandB project name

Example:
  python train_q2_a2.py --epochs 20 --learning_rate 0.0005 --optimizer sgd

----------------------------------------------------------

4. inference_q4_a2.py - Final Evaluation
----------------------------------------
Arguments:
  --model_path best_model.pth  # REQUIRED: Trained model
  --dataset_root ./inaturalist_12K
  --batch_size 8               # Inference batch size

Example:
  python inference_q4_a2.py --model_path ./checkpoints/best_model.pth

==========================================================
Common Issues & Solutions
==========================================================

1. Dataset Not Found:
---------------------
Error: "Training directory not found at: ./inaturalist_12K/train"
Fix:
- Verify dataset is in working directory
- Use absolute path: --dataset_root /full/path/to/inaturalist_12K

2. CUDA Out of Memory:
----------------------
Reduce batch size: --batch_size 16

3. WandB Authentication:
------------------------
wandb login  # Follow on-screen instructions

==========================================================
File Structure
==========================================================

assignment/
├── model_q1_a2.py          # Question 1: Model definition
├── dataset_q2_a2.py        # Question 2: Data loading
├── train_q2_a2.py          # Question 2-3: Training loop
├── inference_q4_a2.py      # Question 4: Evaluation
├── checkpoints/            # For model saving
└── inaturalist_12K/        # Dataset (MUST be placed here)
    ├── train/              # Training images
    └── test/               # Test images

Note: All paths are relative to your execution directory. Run scripts from the directory containing inaturalist_12K/.

==========================================================

