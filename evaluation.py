import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

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
    # Choose device: GPU if available; otherwise, use CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pretrained ResNet-18 model.
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model = model.to(device)

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

    # Load the ImageNet validation set.
    imagenet_val_path = "imagenet_val"
    val_dataset = datasets.ImageFolder(imagenet_val_path, transform=transform)

    # Evaluate the model using the alternative accuracy calculation.
    evaluate(model, val_dataset, batch_size=64, workers=4)
