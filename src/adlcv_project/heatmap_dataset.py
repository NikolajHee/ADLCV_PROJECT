import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class HeatmapDataset(Dataset):
    def __init__(self, index_path, data_root, class_embedding_path, img_size=512):
        with open(index_path, "r") as f:
            self.index = json.load(f)

        self.data_root = data_root
        self.img_size = img_size
        self.class_embeddings = torch.load(class_embedding_path)

    def __len__(self):
        return len(self.index)

    def center_crop_512(self, img):
        w, h = img.size
        left = (w - self.img_size) // 2
        top = (h - self.img_size) // 2
        return img.crop((left, top, left + self.img_size, top + self.img_size))

    def __getitem__(self, idx):
        item = self.index[idx]

        img_path = os.path.join(self.data_root, item["bg_path"])
        img = Image.open(img_path).convert("RGB")
        img = self.center_crop_512(img)

        x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        data = np.load(item["target_path"])
        target = torch.from_numpy(data["target"]).float()  # [8, 32, 32]

        fg_class = item["fg_class"]
        class_embed = self.class_embeddings[fg_class].float()

        return x, class_embed, target, fg_class