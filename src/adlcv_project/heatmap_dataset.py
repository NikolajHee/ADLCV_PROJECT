import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as T


class HeatmapDataset(Dataset):
    def __init__(
        self,
        index_path,
        data_root,
        class_embedding_path,
        img_size=512,
        augment=False,
    ):
        with open(index_path, "r") as f:
            self.index = json.load(f)

        self.data_root = data_root
        self.img_size = img_size
        self.class_embeddings = torch.load(class_embedding_path)

        self.augment = augment

        self.image_augment = T.Compose([
            T.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02,
            ),
            T.RandomGrayscale(p=0.05),
            T.RandomApply([
                T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))
            ], p=0.10),
        ])

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

        if self.augment:
            img = self.image_augment(img)

        x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        data = np.load(item["target_path"])

        target_fg_class = str(data["fg_class"])
        target_bg_path = str(data["bg_path"])

        if target_fg_class != item["fg_class"]:
            print("FG CLASS MISMATCH")
            print("index fg_class:", item["fg_class"])
            print("npz fg_class:", target_fg_class)

        if target_bg_path != item["bg_path"]:
            print("BG PATH MISMATCH")
            print("index bg_path:", item["bg_path"])
            print("npz bg_path:", target_bg_path)

        target = torch.from_numpy(data["target"]).float()

        fg_class = item["fg_class"]
        class_embed = self.class_embeddings[fg_class].float()

        return x, class_embed, target, fg_class, item["bg_path"]

if __name__ == "__main__":
    dataset = HeatmapDataset(
        index_path="data/preprocessed_targets/train/index.json",
        data_root="data",
        class_embedding_path="data/preprocessed_targets/class_embeddings.pt"
    )

    for i in range(10):
        item = dataset.index[i]
        data = np.load(item["target_path"])

        print("------")
        print("index bg_path:", item["bg_path"])
        print("npz bg_path:  ", str(data["bg_path"]))
        print("index class:  ", item["fg_class"])
        print("npz class:    ", str(data["fg_class"]))