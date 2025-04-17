
# 📦 Part A — DA6401 Assignment 2

This repository contains four modular Python scripts to solve Part A of the Deep Learning Assignment 2 (DA6401, IIT Madras). It includes model design, data loading with augmentation, training using Weights & Biases (W&B) sweeps, and final evaluation.

---

## 📁 Folder Structure & Dataset Placement

```
Assignment2/
├── inaturalist_12K/        # Dataset folder (REQUIRED: must contain 'train/' and 'test/')
│   ├── train/
│   └── test/
└── part_a/
    ├── dataset_q2_a2.py
    ├── model_q1_a2.py
    ├── train_sweep_q2_q3_a2.py
    ├── test_q4_a2.py
    └── README.md
```

### 📌 IMPORTANT:

- If the `inaturalist_12K` dataset is placed **inside `part_a/`**, the scripts will detect it automatically using default paths.
- If placed **outside** (e.g., in `Assignment2/`), you **must specify the dataset path** using the `--DATASET_ROOT` argument.
- The dataset must contain `train/` and `test/` folders structured in class-wise subdirectories.

---

## ⚙️ Environment Setup (Tested with CUDA GPU)

Use the following commands to create a reproducible environment:

```bash
conda create -n torch_gpu python=3.9
conda activate torch_gpu

pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install scikit-learn==1.0.2
pip install wandb==0.12.21
pip install numpy==1.21.6
pip install tqdm==4.62.3
pip install thop==0.0.31.post2005241907
pip install matplotlib==3.5.3
```

### ➕ Additional Imports

- Python standard: `os`, `argparse`, `sys`, `traceback`, `types`, `getpass`
- All included by default in Python ≥ 3.6

---

## 🚀 How to Run Scripts from Terminal

### ✅ Always run from `Assignment2/` using this format:

```bash
python part_a/<script_name.py> [--arguments]
```

Examples:
```bash
python part_a/model_q1_a2.py
python part_a/test_q4_a2.py --MODEL_PATH part_a/saved_model.pth --DATASET_ROOT inaturalist_12K
```

---

## 🧠 Script Descriptions

### 1. `model_q1_a2.py` — CNN Model with BatchNorm/Dropout

**Run**:
```bash
python part_a/model_q1_a2.py
```

**Argparse options**:
- `--num_classes`
- `--input_height`
- `--input_width`
- `--input_channels`
- `--filter_counts`
- `--kernel_size`
- `--activation`
- `--dense_neurons`
- `--pool_kernel_size`
- `--pool_stride`
- `--dropout_rate`
- `--use_batchnorm`

---

### 2. `dataset_q2_a2.py` — Data Loaders with Stratified Split

**Run**:
```bash
python part_a/dataset_q2_a2.py --DATASET_ROOT inaturalist_12K
```

**Argparse options**:
- `--DATASET_ROOT`

---

### 3. `train_sweep_q2_q3_a2.py` — W&B Sweep Training

**⚠️ NOTE**: You **must specify your W&B API key** via:
```bash
--WANDB_API_KEY <your_api_key>
```

**Run**:
```bash
python part_a/train_sweep_q2_q3_a2.py --WANDB_API_KEY <your_api_key>
```

**Argparse options**:
- `--SEED`
- `--DATASET_ROOT`
- `--IMAGE_SIZE`
- `--WANDB_ENTITY`
- `--WANDB_PROJECT`
- `--NEW_SWEEP_RUN_COUNT`
- `--WANDB_API_KEY`

---

### 4. `test_q4_a2.py` — Evaluation & W&B Prediction Grid

**Run**:
```bash
python part_a/test_q4_a2.py --MODEL_PATH part_a/saved_model.pth --DATASET_ROOT inaturalist_12K
```

**Argparse options**:
- `--MODEL_PATH` *(required)*
- `--DATASET_ROOT`
- `--BATCH_SIZE`
- `--NUM_CLASSES`

---

## 🖼️ WandB Output (Q3, Q4)

- **Q3**: Each sweep trial logs training and validation metrics
- **Q4**: A 10×3 image table is logged showing predicted vs true classes (3 images per class)

---

## ✍️ Author

> **DA6401 - Deep Learning**  
> *A B Keshav Kumar (AE24S021),MS Scholar, IIT Madras*  
> *Assignment 2 - Part A*

---

## 💬 Need Help?

If any script fails due to import/module issues, check:
- Python version (3.9 recommended)
- CUDA 11.3 required for GPU support
- Dataset path structure
- W&B login status (`wandb login`)
