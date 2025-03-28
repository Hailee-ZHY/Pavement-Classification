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
        self.num_classes = max(self.label2id.values())+1

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

        # for valuation part
        self.best_val_loss = float('inf') # update with smaller loss until get the smallet one

    def train_step(self, epoch):
        self.model.train()
        total_loss = 0 

        # for images, mask in tqdm(self.train_loader, desc = f"Epoch {epoch+1}/{self.cfg.num_epoch}"):
        # debug 
        for batch in tqdm(self.train_loader, desc = f"Epoch {epoch+1}/{self.cfg.num_epoch}"):
            if batch is None or not isinstance(batch, (list, tuple)) or len(batch) != 2: # 跑完周感觉这里可以改一下，应该也没问题
                print("Skipped invalid batch.")
                continue 
            
            images, masks = batch

            images = images.to(self.device)
            masks = masks.to(self.device)

            outputs = self.model(pixel_values = images)
            logits = outputs.logits #[B,num_class,H,W] 
            # resize logits to match with masks
            logits = torch.nn.functional.interpolate(
                logits, size = masks.shape[-2:], mode = "bilinear", align_corners=False
            )

            # print(f"[DEBUG] logits shape: {logits.shape}")
            # print(f"[DEBUG] masks shape: {masks.shape}, dtype: {masks.dtype}")
            # print(f"[DEBUG] masks unique: {masks.unique()}")
            # print(f"[DEBUG] masks min: {masks.min().item()}, max: {masks.max().item()}, num_classes: {self.num_classes}")


            loss = self.criterion(logits, masks)
            self.optimizer.zero_grad()

            # # debug
            # if not torch.isfinite(loss):
            #     print(f"[ERROR] non-finite loss: {loss}")
            #     continue

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        print(f"Epoch {epoch +1} - Training Loss: {avg_loss:.4f}")

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_pixel = 0
        num_classes = self.num_classes
        intersection = torch.zeros(num_classes, device = self.device)
        union = torch.zeros(num_classes, device = self.device)

        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                output = self.model(pixel_values = images)
                logits = output.logits
                # resize, same as train part
                logits = torch.nn.functional.interpolate(
                    logits, size = masks.shape[-2:], mode = "bilinear", align_corners=False
                    )
                
                loss = self.criterion(logits, masks)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim = 1)
                total_correct += (preds == masks).sum().item()
                total_pixel += torch.numel(masks)

                for cls in range(num_classes):
                    pred_inds = (preds == cls)
                    target_inds = (masks == cls)
                    intersection[cls] += (pred_inds & target_inds).sum()
                    union[cls] += (pred_inds | target_inds).sum()

        avg_loss = total_loss /len(self.val_loader)
        pixel_acc = total_correct / total_pixel
        iou = (intersection/(union + 1e-6)).mean().item()

        print(f"Validation Loss: {avg_loss: 4f}, Pixel Acc: {pixel_acc:4f}, mIoU:{iou: 4f}")
        return avg_loss
        
    
    def run(self):
        print("start training ...")
        for epoch in range(self.cfg.num_epoch):
            self.train_step(epoch)
            val_loss = self.evaluate()

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_model.pth")
        print("End training, new best model weight saving ...")
    
if __name__ == "__main__":
    cfg = segformerConfig()
    trainer = SegformerTrainer(cfg)
    trainer.run()


