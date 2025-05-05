#!/bin/bash

# Train the model
echo "Training..."
python train.py

# Inference on the test dataset
echo "Inferencing..."
python inference.py --model_path best_model_20250330_1857.pth # this is the file with the best weight of mine, you can change it into yours 