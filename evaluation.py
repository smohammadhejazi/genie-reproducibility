import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
from torch.utils.data import Subset
from reconstruct import QuantizableLayer

def evaluate(model, dataset, batch_size=64, workers=4):
    model.eval()  # Set the model to evaluation mode
    device = next(model.parameters()).device  # Detect which device the model is on (CPU/GPU)

    # Create a DataLoader for the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, pin_memory=True)

    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient tracking for evaluation
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)  # Forward pass
            preds = outputs.argmax(dim=1)  # Get predicted class labels

            # Move predictions and labels to CPU and store them
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            # Print accuracy for current batch
            batch_accuracy = (preds.cpu() == labels).float().mean().item()
            print(f"Batch accuracy: {batch_accuracy * 100:.2f}%")

    # Combine all batches into single tensors
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Calculate final accuracy across the whole dataset
    accuracy = (all_preds == all_labels).float().mean().item()
    print(f"Top-1 Accuracy: {accuracy * 100:.2f}%")

    return accuracy

if __name__ == "__main__":
    # Define standard ImageNet preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize(256),               # Resize shorter side to 256 pixels
        transforms.CenterCrop(224),           # Center crop to 224x224
        transforms.ToTensor(),                # Convert to PyTorch tensor
        transforms.Normalize(                 # Normalize using ImageNet statistics
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Load ImageNet validation set (or a custom dataset in 'imagenet_val' folder)
    imagenet_val_path = "imagenet_val"
    val_dataset = datasets.ImageFolder(imagenet_val_path, transform=transform)

    # Randomly sample a subset of 512 images for quick evaluation
    subset_size = 512
    all_indices = torch.randperm(len(val_dataset))[:subset_size]
    subset_indices = all_indices.tolist()
    val_dataset = Subset(val_dataset, subset_indices)

    # Choose GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------
    # Evaluate standard ResNet-18
    # ---------------------------
    print("Evaluating resnet18")
    model1 = models.resnet18(weights=ResNet18_Weights.DEFAULT)  # Load pretrained weights
    model1 = model1.to(device)
    model1.eval()
    evaluate(model1, val_dataset, batch_size=64, workers=4)

    # -----------------------------------
    # Evaluate quantized ResNet-18 model
    # -----------------------------------
    print("Evaluating resnet18_quantized")
    model2 = torch.load('saved_models\\resnet18_quantized.pth', weights_only=False)
    model2 = model2.to(device)
    model2.eval()
    evaluate(model2, val_dataset, batch_size=64, workers=4)

    # -----------------------------------------
    # Evaluate quantized + optimized ResNet-18
    # -----------------------------------------
    print("Evaluating resnet18_quantized_optimized")
    model3 = torch.load('saved_models\\resnet18_quantized_optimized.pth', weights_only=False)
    model3 = model3.to(device)
    model3.eval()
    evaluate(model3, val_dataset, batch_size=64, workers=4)

