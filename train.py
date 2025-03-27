"""
segmentation -> instance extraction -> classification
Segmentation Models: SegFormer | Swin-Unet | DeepLabV3/DeepLabV3++
Classification Models: ViT
"""

import torch 
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation
from tqdm import tqdm 
import json
import os

from DatasetProcessor import RoadMarkingDataset
from config import segformerConfig

class SegformerTrainer:
    def __init__(self, config):
        self.cfg = config
        self.device = config.device

        # load label2id and class count
        with open(self.cfg.label_map_path, "r") as f:
            self.label2id = json.load(f)
        self.num_classes = len(self.label2id)

        # initialize dataset and dataloaders 
        self.train_dataset = RoadMarkingDataset(self.cfg.image_dir, self.cfg.mask_dir, transform = True, split_file=self.cfg.split_file)
        self.val_dataset = RoadMarkingDataset(self.cfg.image_dir, self.cfg.mask_dir, transform = False, split_file=self.cfg.val_split_file)

        self.train_loader = DataLoader(self.train_dataset, batch_size = self.cfg.batch_size, shuffle = True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.cfg.batch_size, shuffle = False)

        # initialize model 
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.cfg.model_name, 
            num_labels = self.num_classes, 
            ignore_mismatched_sizes=True, 
        ).to(self.device)

        # loss and optimizer 
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate)

    def train_step(self, epoch):
        self.model.train()
        total_loss = 0 

        # for images, mask in tqdm(self.train_loader, desc = f"Epoch {epoch+1}/{self.cfg.num_epoch}"):
        # debug 
        for batch in tqdm(self.train_loader, desc = f"Epoch {epoch+1}/{self.cfg.num_epoch}"):
            if batch is None or not isinstance(batch, (list, tuple)) or len(batch) != 2:
                print("Skipped invalid batch.")
                continue 
            
            images, masks = batch

            images = images.to(self.device)
            masks = masks.to(self.device)

            outputs = self.model(pixel_values = images)
            logits = outputs.logits #[B,C,H,W]

            loss = self.criterion(logits, masks)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimize.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        print(f"Epoch {epoch +1} - Training Loss: {avg_loss:.4f}")

    def run(self):
        print("start training ...")
        for epoch in range(self.cfg.num_epoch):
            self.train_step(epoch)
    
if __name__ == "__main__":
    cfg = segformerConfig()
    trainer = SegformerTrainer(cfg)
    trainer.run()


