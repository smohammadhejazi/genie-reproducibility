import torch
import torchvision.models as models

resnet18 = models.resnet18(pretrained=True)
input_tensor = torch.randn(1, 3, 224, 224)
output = resnet18(input_tensor)
print(output)

mobilenet_v2 = models.mobilenet_v2(pretrained=True)
input_tensor = torch.randn(1, 3, 224, 224)
output = mobilenet_v2(input_tensor)
print(output)