#!/bin/bash

# Train the model
echo "Training..."
python train.py

# Inference on the test dataset
echo "Inferencing..."
python inference.py --model_path best_model.pth