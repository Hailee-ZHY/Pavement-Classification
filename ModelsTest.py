# model test result & visualization
# current models: SegTranformer

from config import segformerConfig
from DatasetProcessor import RoadMarkingDataset
import json
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation
from config import segformerConfig
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np

class SegformerTester:
    def __init__(self, config):
        self.cfg = config
        self.device = config.device

        with open(self.cfg.label_map_path, "r") as f:
            self.label2id = json.load(f)
        
        self.num_classes = max(self.label2id.values()) + 1 

        self.test_dataset = RoadMarkingDataset(
            image_dir=self.cfg.image_dir, 
            mask_dir = self.cfg.mask_dir, 
            transform = False,
            split_file=self.cfg.test_split_file
        )
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle = False)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.cfg.model_name,
            num_labels = self.num_classes, 
            ignore_mismatched_sizes=True,
        ).to(self.device)

        # load best weight
        self.model.load_state_dict(torch.load("best_model.pth", map_location=self.device))
        self.model.eval()
    
    def visualization_test_result(self, num_visualize = 5):
        count = 0
        with torch.no_grad():
            for images, masks in tqdm(self.test_loader, desc = "Testing"):
                images = images.to(self.device)
                masks = masks.to(self.device)

                output = self.model(pixel_values = images)
                logits = output.logits
                # resize, same as train part
                logits = torch.nn.functional.interpolate(
                    logits, size = masks.shape[-2:], mode = "bilinear", align_corners=False
                    )
                
                preds = torch.argmax(logits, dim = 1) # [B,C,H,W], 选择概率最大对应的Class

                # Visualization 
                if count < num_visualize:
                    image_np = images[0].permute(1,2,0).cpu().numpy() # 提取都是当前batch的第一张图，而且batch_size设置的为1
                    if image_np.max() <= 1.0: # 说明是经过ToTensor归一化了
                        image_np = (image_np * 255).astype(np.uint8)
                    mask_np = masks[0].cpu().numpy()
                    pred_np = preds[0].cpu().numpy()
                    print(f"[debug] Prediction unique values: {np.unique(pred_np)}")

                    fig, axes = plt.subplots(1,3,figsize = (15,5))
                    axes[0].imshow(image_np.astype(np.uint8))
                    axes[0].set_title("Image")
                    axes[1].imshow(mask_np, cmap = "nipy_spectral")  # cmap = 指定colormap
                    axes[1].set_title("Ground Truth")
                    axes[2].imshow(pred_np, cmap = "nipy_spectral")
                    axes[2].set_title("Prediction")
                    for ax in axes:
                        ax.axis("off")
                    plt.tight_layout
                    plt.show()
                
                count += 1

if __name__ == "__main__":
    cfg = segformerConfig()
    tester = SegformerTester(cfg)
    tester.visualization_test_result(num_visualize=5)






