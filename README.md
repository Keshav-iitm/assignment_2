# 📦 DA6401 Assignment 2 _ A B Keshav Kumar (AE24S021)


# 🔗 [GitHub Repository] https://github.com/Keshav-iitm/assignment_2.git
# 📊 [WandB Project Link] https://wandb.ai/ae24s021-indian-institute-of-technology-madras/DA6401-Assignment2/reports/Assignment-2-A-B-Keshav-Kumar-AE24S021-DA6401--VmlldzoxMjIxNzU4OQ?accessToken=a9glsz1iqr0jii0ykwiaw7f8ywh946j56xjojv9fjuq3pnlqhd758uj70gncmigh


This repository contains modular Python scripts to solve Part A and Part B of the Deep Learning Assignment 2 (DA6401, IIT Madras). It includes model design, training with hyperparameter sweeps using Weights & Biases (W&B),testing it and fine-tuning a pretrained model on iNaturalist.

---

## 📁 Folder Structure & Dataset Placement

```
Assignment2/
├── inaturalist_12K/        # Dataset folder (REQUIRED: must contain 'train/' and 'test/')
│   ├── train/
│   └── test/
├──README.md
├── part_a/
│   ├── dataset_q2_a2.py
│   ├── model_q1_a2.py
│   ├── train_sweep_q2_q3_a2.py
│   ├── test_q4_a2.py
│  
└── part_b/
    ├── fine_tune.py        # Fine-tunes a ResNet50 model on iNaturalist
```

### 📌 IMPORTANT:

- If the `inaturalist_12K` dataset is placed **inside `part_a/`**, Part A scripts will detect it automatically.
- If placed **anywhere else (e.g., in parent folder)**, just use `--DATASET_ROOT` argument (needed in **Part A only**).
- **Part B works directly** as long as dataset is accessible at the specified root. It does **not require** the dataset to be inside `part_b/`.

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
python part_b/fine_tune.py [--arguments]
```

---

## 🧠 Script Descriptions

### ✅ Part A Scripts
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
- `--DATASET_ROOT` #should be specified if inaturalist_12K is not in part_a folder

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
- `--DATASET_ROOT`#should be specified if inaturalist_12K is not in part_a folder
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
- `--DATASET_ROOT`#should be specified if inaturalist_12K is not in part_a folder
- `--BATCH_SIZE`
- `--NUM_CLASSES`

---

### ✅ Part B Script — Fine-Tuning ResNet50 on iNaturalist

**File**: `part_b/fine_tune.py`  
**Description**: This script fine-tunes a pre-trained ResNet50 model on the inaturalist_12K dataset.

**Run**:
```bash
python part_b/fine_tune.py
```

**Argparse options**:
- `--dataset_root` : Path to the dataset folder (default: `./inaturalist_12K`)
- `--batch_size` : Batch size for training (default: `32`)
- `--epochs` : Number of training epochs (default: `10`)
- `--lr` : Learning rate for optimizer (default: `0.001`)

**Features**:
- Uses torchvision’s `ResNet50` pretrained on ImageNet
- Freezes all layers except final classification head
- Automatically handles GPU/CPU
- Prints training stats and final validation accuracy

---

## 🖼️ WandB Output (Q3, Q4)

- **Q3 (Part A)**: Each sweep trial logs training and validation metrics
- **Q4 (Part A)**: A 10×3 image table is logged showing predicted vs true classes (3 images per class)

---

## ✍️ Author

> **DA6401 - Deep Learning**  
>  *A B Keshav Kumar (AE24S021),MS Scholar, IIT Madras* 
> *Assignment 2 - Part A & Part B*

---

## 💬 Need Help?

If any script fails due to import/module issues, check:
- Python version (3.9 recommended)
- CUDA 11.3 required for GPU support
- Dataset path structure
- W&B login status (`wandb login`)
