import torch
import torchvision.models as models

resnet18 = models.resnet18(pretrained=True)
print(resnet18)

input_tensor = torch.randn(1, 3, 224, 224)
output = resnet18(input_tensor)
print(output)
