import argparse
import sys
import os
import torch
from scripts.train import train_model
from scripts.test import test_model

def main():
    parser = argparse.ArgumentParser(description="DeWatermark Project")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                        help="Choose train or test mode")
    parser.add_argument("--dir", type=str, default="", help="Directory of clean images")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--model_path", type=str, default="models/generator_epoch_100.pth", help="Path to the trained model")
    parser.add_argument("--pretrained", type=str, default="", help="Use pre-trained model")
    parser.add_argument("--input_image", type=str, help="Path to input image for testing (optional for batch mode)")
    parser.add_argument("--output_folder", type=str, help="Path to save output image (optional for batch mode)")

    args = parser.parse_args()

    if args.mode == "train":
        train_model(epochs=args.epochs, dir=args.dir, pretrained_pth=args.pretrained)
    elif args.mode == "test":
        if args.input_image and args.output_folder:
            test_model(model_path=args.model_path, input_image_path=args.input_image, output_folder=args.output_folder)
        else:
            test_model(model_path=args.model_path)

if __name__ == "__main__":
    main()
