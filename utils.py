# Add all helper functions here

import os 
import random 

def load_split_file(split_file):
    with open(split_file, "r") as f:
        filenames = [line.strip() for line in f if line.strip()]
    return filenames

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
