from pathlib import Path

import typer
from torch.utils.data import Dataset

from loguru import logger
import torchvision.datasets as datasets

import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torchvision.transforms as T

app = typer.Typer()

class HiddenObjectsDataset(Dataset):
    def __init__(self, places_root, split="train"):
        self.hf_data = load_dataset("marco-schouten/hidden-objects", split=split)
        self.places_root = places_root
        self.transform = T.Compose([
            T.Resize(512),
            T.CenterCrop(512),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.hf_data)

    def __getitem__(self, idx):
        item = self.hf_data[idx]
        img_path = os.path.join(self.places_root, item['bg_path'])
        image = self.transform(Image.open(img_path).convert("RGB"))
        bbox = torch.tensor(item['bbox']) * 512
        return {
            "image": image,
            "bbox": bbox,
            "label": item['label'],
            "class": item['fg_class'],
            "image_reward_score" : item['image_reward_score'],
            "confidence" : item['confidence']}
class HiddenObjectsDatasetStreaming(HiddenObjectsDataset):
    def __getitem__(self, idx):
        item = self.hf_data[idx]
        img_path = os.path.join(self.places_root, item['bg_path'])
        image = self.transform(Image.open(img_path).convert("RGB"))
        bbox = torch.tensor(item['bbox']) * 512
        return {
            "entry_id": item['entry_id'],
            "image": image, 
            "bbox": bbox, 
            "label": item['label'], 
            "class": item['fg_class']
        }

def get_streaming_loader(places_root, batch_size=32):
    dataset = load_dataset("marco-schouten/hidden-objects", split="train", streaming=True)
    preprocess = T.Compose([T.Resize(512), T.CenterCrop(512), T.ToTensor()])

    def collate_fn(batch):
        images, bboxes, ids = [], [], []
        for item in batch:
            path = os.path.join(places_root, item['bg_path'])
            try:
                img = Image.open(path).convert("RGB")
                images.append(preprocess(img))
                bboxes.append(torch.tensor(item['bbox']) * 512)
                ids.append(item['entry_id'])
            except FileNotFoundError:
                continue
        return {
            "entry_id": ids,
            "pixel_values": torch.stack(images), 
            "bboxes": torch.stack(bboxes)
        }
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

@app.command()
def download_background_images() -> None:
    root = "data/places365"   
    dataset = datasets.Places365(root=root, split='val', small=True, download=True)
    logger.info(f"Downloaded background images: {len(dataset)} to {root}")

@app.command()
def other() -> None:
    root = "data/places365"   
    dataset = datasets.Places365(root=root, split='val', small=True, download=True)
    logger.info(f"Downloaded background images: {len(dataset)} to {root}")


if __name__ == "__main__":
    app()
