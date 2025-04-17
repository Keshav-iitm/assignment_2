
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from model_q1_a2 import CNNModel
from dataset_q2_a2 import get_dataloaders
import argparse
import os
from collections import defaultdict

# Configuration
def get_args():
    parser = argparse.ArgumentParser(description="Model Inference/Config Args")
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_DATASET_ROOT = os.path.join(SCRIPT_DIR, "inaturalist_12K")
    parser.add_argument("--DATASET_ROOT", type=str, default=DEFAULT_DATASET_ROOT,
                        help="Path to the dataset folder (default: ./inaturalist_12K relative to script)")
    parser.add_argument("--BATCH_SIZE", type=int, default=8,
                        help="Batch size for inference/training")
    parser.add_argument("--MODEL_PATH", type=str, required=True,
                        help="(Required) Path to the saved model file")
    parser.add_argument("--NUM_CLASSES", type=int, default=10,
                        help="Number of classes in the dataset")
    return parser.parse_args()

# Unnormalizing helper
def unnormalize(img_tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.numpy()
    img = std[:, None, None] * img + mean[:, None, None]
    img = np.clip(img, 0, 1)
    return np.transpose(img, (1, 2, 0))

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
        "use_batchnorm": True
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel(**model_params)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    _, _, test_loader, _, class_names = get_dataloaders(
        dataset_root=DATASET_ROOT,
        batch_size=BATCH_SIZE,
        use_augmentation=False
    )
    if not class_names:
        class_names = [str(i) for i in range(NUM_CLASSES)]

    correct, total = 0, 0
    class_to_images = defaultdict(list)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            for img, pred, true in zip(images, predicted, labels):
                true_cls = true.item()
                if len(class_to_images[true_cls]) < 3:
                    class_to_images[true_cls].append((img.cpu(), pred.item(), true_cls))
            if all(len(v) == 3 for v in class_to_images.values()) and len(class_to_images) == NUM_CLASSES:
                break

    print(f"Test Accuracy: {correct / total:.4f}")

    wandb.init(project="DA6401-Assignment2", name="10x3_test_grid")

    columns = ["image", "predicted", "true", "correct"]
    table = wandb.Table(columns=columns)

    for cls in range(NUM_CLASSES):
        for img, pred, true in class_to_images.get(cls, []):
            img_np = (unnormalize(img) * 255).astype(np.uint8)
            pred_name = class_names[pred]
            true_name = class_names[true]
            emoji = "✅" if pred == true else "❌"
            table.add_data(wandb.Image(img_np), pred_name, true_name, emoji)

    wandb.log({"10x3_test_grid": table})
    wandb.finish()

    all_images = []
    for cls_imgs in class_to_images.values():
        all_images.extend(cls_imgs)

    fig, axes = plt.subplots(NUM_CLASSES, 3, figsize=(9, NUM_CLASSES * 3))
    fig.suptitle('10×3 Grid of Test Images (Each Row: 1 Class)', fontsize=20)
    for i, ax in enumerate(axes.flat):
        if i >= len(all_images):
            ax.axis('off')
            continue
        img, pred, true = all_images[i]
        img_np = unnormalize(img)
        color = 'green' if pred == true else 'red'
        ax.imshow(img_np)
        ax.axis('off')
        ax.set_title(f"Pred: {class_names[pred]}\nTrue: {class_names[true]}", fontsize=9, color=color)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
