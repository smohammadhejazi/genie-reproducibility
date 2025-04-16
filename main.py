import sys
import argparse
import torchvision.models as models
from torchvision.models import ResNet18_Weights, MobileNet_V2_Weights
from distil import distill_data
from reconstruct import quantize_model, reconstruct
from evaluation import evaluate


def main():
    # Simulate command-line arguments for testing/debugging
    # sys.argv = ["main.py", "resnet18", "-d"]
    
    # Set program name and description for argument parser
    prog = "Genie Reproducibility Challenge"
    descr = "Distill data and quantize models"
    
    # Initialize the argument parser with a custom program name and description
    parser = argparse.ArgumentParser(prog=prog, description=descr)
    
    # Define optional flags for different stages of the model processing pipeline
    parser.add_argument("-d", "--distill", action="store_true", help="Distill images")
    parser.add_argument("-q", "--quantize", action="store_true", help="Quantize model")
    parser.add_argument("-r", "--reconstruct", action="store_true", help="Reconstruct quantized model using distillation")
    parser.add_argument("-e", "--evaluate", action="store_true", help="Evaluate model")
    
    # Define a required positional argument for selecting the model
    parser.add_argument("model", choices=["resnet18", "mobilenet_v2"], help="Model to use in distill and quantize")
    
    # Parse command-line arguments
    args = parser.parse_args()

    # Load the selected pretrained model with default weights
    if args.model == "resnet18":
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    elif args.model == "mobilenet_v2":
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

    # Perform dataset distillation if requested
    if args.distill:
        train_dataset = distill_data(model)

    # Perform model quantization if requested
    # Done in the jupyter notebook python file reconstruction.ipynb
    # if args.quantize:
    #     quantized_model = quantize_model(model)

    # Perform reconstruction using the distilled dataset and quantized model if requested
    # Done in the jupyter notebook python file reconstruction.ipynb
    # if args.reconstruct:
    #     reconstruct(model, quantized_model, train_dataset)

    # Evaluate the quantized model if requested
    if args.evaluate:
        evaluate(model)

# Execute the main function when the script is run directly
if __name__ == "__main__":
    main()
