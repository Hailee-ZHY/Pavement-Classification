#!/bin/bash

# prepare dataset
python dataset_loader.py
echo "split dataset into train/eval/test..."
python split_dataset.py