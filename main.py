import torch
import sys
import argparse
import torchvision.models as models
from torchvision.models import ResNet18_Weights, MobileNet_V2_Weights
from distil import distill_data
from quantize import quantize_model


def main():

    sys.argv = ["main.py", "resnet18", "-d", "-q"]
    
    prog = "Genie Reproducibility Challenge"
    descr = "Distill data and quantize models"
    parser = argparse.ArgumentParser(prog=prog, description=descr)
    parser.add_argument("-d", "--distill", required=True, action="store_true", help="Distill images")
    parser.add_argument("-q", "--quantize", required=True, action="store_true", help="Quantize model")
    parser.add_argument("model", choices=["resnet18", "mobilenet_v2"], help="Model to use in distill and quantize")
    args = parser.parse_args()

    if args.model == "resnet18":
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    elif args.model == "mobilenet_v2":
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

    if args.distill:
        distill_data(model)
    if args.quantize:
        quantize_model(model)


if __name__ == "__main__":
    main()
