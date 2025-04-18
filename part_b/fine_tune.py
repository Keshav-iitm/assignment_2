import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader

def get_args():
    parser = argparse.ArgumentParser(description="Fine-tune pre-trained model on iNaturalist")
    parser.add_argument("--dataset_root", type=str, default="./inaturalist_12K",
                       help="Path to dataset (default: ./inaturalist_12K)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training (default: 32)")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate (default: 0.001)")
    return parser.parse_args()

def main():
    args = get_args()
    
    # Verify dataset path exists
    if not os.path.isdir(args.dataset_root):
        raise FileNotFoundError(f"Dataset directory not found at: {args.dataset_root}")

    # Configure device (GPU/CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Define image transformations for ResNet50
    transform = transforms.Compose([
        transforms.Resize((224, 224)),          # Resize to match ResNet input
        transforms.ToTensor(),                  #Convert to tensor
        transforms.Normalize(                   # ImageNet normalization
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load training and validation datasets
    train_dataset = datasets.ImageFolder(
        root=os.path.join(args.dataset_root, 'train'),
        transform=transform
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(args.dataset_root, 'test'),
        transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize pre-trained ResNet50 with ImageNet weights
    model = models.resnet50(pretrained=True)
    
    # Freeze all layers except the final one
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(train_dataset.classes))  # 10 classes
    
    model = model.to(device)  # Move model to device

    # Print model configuration
    print("Model configuration:")
    print(f"- Pre-trained weights: ImageNet")
    print(f"- Final layer: {model.fc}")
    print(f"- Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"- Training samples: {len(train_dataset)}")
    print(f"- Validation samples: {len(val_dataset)}\n")

    # Loss function and optimizer (only for the final layer)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    # Training loop
    print(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()  # Clear gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            
            # Track metrics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Print epoch statistics
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    # Validation after training
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"\nValidation Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    main()
