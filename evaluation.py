import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
from torch.utils.data import Subset
from reconstruct import QuantizableLayer

def evaluate(model, dataset, batch_size=64, workers=4):
    model.eval()
    device = next(model.parameters()).device
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, pin_memory=True)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            # Accumulate predictions and labels, moving them to CPU for concatenation.
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            batch_accuracy = (preds.cpu() == labels).float().mean().item()
            print(f"Batch accuracy: {batch_accuracy * 100:.2f}%")

    # Concatenate all predictions and labels from the batches.
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # Compute overall accuracy by comparing concatenated tensors.
    accuracy = (all_preds == all_labels).float().mean().item()
    print(f"Top-1 Accuracy: {accuracy * 100:.2f}%")
    return accuracy

if __name__ == "__main__":
    # Define the transforms for ImageNet validation.
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean values.
            std=[0.229, 0.224, 0.225]    # ImageNet standard deviation.
        ),
    ])

    # Select a subset
    imagenet_val_path = "imagenet_val"
    val_dataset = datasets.ImageFolder(imagenet_val_path, transform=transform)
    subset_size = 512
    all_indices = torch.randperm(len(val_dataset))[:subset_size]
    subset_indices = all_indices.tolist()
    val_dataset = Subset(val_dataset, subset_indices)

    # Choose device: GPU if available; otherwise, use CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pretrained ResNet-18 model.
    print("Evaluting resnet18")
    model1 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model1 = model1.to(device)
    model1.eval()
    evaluate(model1, val_dataset, batch_size=64, workers=4)

    print("Evaluting resnet18_quantized")
    model2 = torch.load('saved_models\\resnet18_quantized.pth', weights_only=False)
    model2 = model2.to(device)
    model2.eval()
    evaluate(model2, val_dataset, batch_size=64, workers=4)

    print("Evaluting resnet18_quantized_optimized")
    model3 = torch.load('saved_models\\resnet18_quantized_optimized.pth', weights_only=False)
    model3 = model3.to(device)
    model3.eval()
    evaluate(model3, val_dataset, batch_size=64, workers=4)
