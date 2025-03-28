# Add all helper functions here

import os 
import random 
from collections import Counter
from tqdm import tqdm
import torch

def load_split_file(split_file):
    with open(split_file, "r") as f:
        filenames = [line.strip() for line in f if line.strip()]
    return filenames

def analyze_class_distribution(dataloader):
    counter = Counter()

    for _, mask in tqdm(dataloader, desc="Counting pixels"):
        for m in mask:  # loop over batch dimension
            if isinstance(m, torch.Tensor):
                m = m.cpu().numpy()
            m = m.flatten()
            counter.update(m.tolist())

    print("\nClass Distribution:")
    for class_id, count in sorted(counter.items()):
        print(f"Class {class_id}: {count:,} pixels")

    return counter # A dictionary-like object mapping class indices to pixel counts.

def compute_class_weights(counter, num_classes):
    total = sum(counter.values())
    weights = []
    for i in range(num_classes):
        count = counter.get(i, 0)
        if count == 0:
            weights.append(0.0)  # or small value like 1e-6 to avoid zero
        else:
            weights.append(total / count)

    weights = torch.tensor(weights, dtype=torch.float)
    weights = weights / weights.sum()  # normalize to sum to 1 (optional)
    print("\nClass Weights:", weights.numpy())
    return weights #torch.Tensor: A tensor of class weights for use in loss functions.

class SplitHelper:
    @staticmethod
    # 这个方法用于帮助我们将指定train/val文件中的image和mask pair起来，给RoadMarkingDataset读取
    def get_split_indices(image_dir, split_file):
        """given a split file, return valid image and mask filenames"""
        split_name = load_split_file(split_file)
        image_files = []
        mask_files = []
        for name in split_name:
            image_path = os.path.join(image_dir, name)
            mask_name = name.replace(".png", "_mask.png")
            if os.path.exists(image_path):
                image_files.append(name)
                mask_files.append(mask_name)
            else:
                print(f"skipped missing image: {image_path}")
        return image_files, mask_files

