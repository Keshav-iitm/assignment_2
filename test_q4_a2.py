import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from model_q1_a2 import CNNModel
from dataset_q2_a2 import get_dataloaders
import argparse
import os


# Configuration

def get_args():
    parser = argparse.ArgumentParser(description="Model Inference/Config Args")

    parser.add_argument("--DATASET_ROOT", type=str, default="./inaturalist_12K",
                        help="Path to the dataset folder")
    parser.add_argument("--BATCH_SIZE", type=int, default=8,
                        help="Batch size for inference/training")

    # Required MODEL_PATH
    parser.add_argument("--MODEL_PATH", type=str, required=True,
                        help="(Required) Path to the saved model file")

    parser.add_argument("--NUM_CLASSES", type=int, default=10,
                        help="Number of classes in the dataset")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    DATASET_ROOT = args.DATASET_ROOT
    BATCH_SIZE = args.BATCH_SIZE
    MODEL_PATH = args.MODEL_PATH
    NUM_CLASSES = args.NUM_CLASSES

    print(" Config loaded:")
    print(f"  DATASET_ROOT: {DATASET_ROOT}")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    print(f"  MODEL_PATH: {MODEL_PATH}")
    print(f"  NUM_CLASSES: {NUM_CLASSES}")


model_params = {
    "num_classes": NUM_CLASSES,
    "input_channels": 3,
    "input_height": 224,
    "input_width": 224,
    "filter_counts": [16, 16, 32, 32, 64],  
    "kernel_size": 3,
    "activation_name": "mish",
    "dense_neurons": 256,
    "dropout_rate": 0.5,
    "use_batchnorm": True #Results based on my best run
}

# Loading model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel(**model_params)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Loading Test Data
_, _, test_loader, _, class_names = get_dataloaders(
    dataset_root=DATASET_ROOT,
    batch_size=BATCH_SIZE,
    use_augmentation=False
)
if not class_names:
    class_names = [str(i) for i in range(NUM_CLASSES)]

# Collecting images and loading test accuracy
correct, total = 0, 0
all_images, all_preds, all_trues = [], [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        # For grid: store images and labels (up to 30)
        if len(all_images) < 30:
            for img, pred, true in zip(images, predicted, labels):
                if len(all_images) < 30:
                    all_images.append(img.cpu())
                    all_preds.append(pred.cpu().item())
                    all_trues.append(true.cpu().item())
print(f"Test Accuracy: {correct / total:.4f}")

# Unnormalising 
def unnormalize(img_tensor):
    import numpy as np
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.numpy()
    img = std[:, None, None] * img + mean[:, None, None]
    img = np.clip(img, 0, 1)
    return np.transpose(img, (1, 2, 0))

# Grid to WANDB
wandb.init(project="DA6401-Assignment2", name="test_image_grid")

columns = ["image", "predicted", "true", "correct"]
table = wandb.Table(columns=columns)

for i in range(len(all_images)):
    img = unnormalize(all_images[i])
    pred = all_preds[i]
    true = all_trues[i]
    pred_name = class_names[pred] if class_names else str(pred)
    true_name = class_names[true] if class_names else str(true)
    correct_emoji = "✅" if pred == true else "❌"
    table.add_data(wandb.Image(img), pred_name, true_name, correct_emoji)

wandb.log({"test_image_grid": table})
wandb.finish()


# (Optional) Local grid
fig, axes = plt.subplots(10, 3, figsize=(9, 30))
fig.suptitle('10×3 Grid of Test Images with Predictions', fontsize=20)
for i, ax in enumerate(axes.flat):
    if i >= len(all_images):
        ax.axis('off')
        continue
    img = unnormalize(all_images[i])
    pred = all_preds[i]
    true = all_trues[i]
    color = 'green' if pred == true else 'red'
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f"Pred: {class_names[pred]}, True: {class_names[true]}", fontsize=10, color=color)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

