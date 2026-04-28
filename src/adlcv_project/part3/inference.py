"""Score bbox placements against your trained CLIP-FiLM heatmap model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from adlcv_project.models.resnet import MultiScaleBackbone
from adlcv_project.models.transformer import SimpleTransformer
from adlcv_project.models.model import MainModel, Decoder, TextEncoder


def center_crop_512(img: Image.Image, img_size: int = 512) -> Image.Image:
    w, h = img.size
    left = (w - img_size) // 2
    top = (h - img_size) // 2
    return img.crop((left, top, left + img_size, top + img_size))


def preprocess_image(image, device, img_size: int = 512) -> torch.Tensor:
    if isinstance(image, Image.Image):
        image = center_crop_512(image.convert("RGB"), img_size)
        image = np.asarray(image)

    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        if image.max() > 1:
            image = image / 255.0

    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            image = image.unsqueeze(0)
        elif image.ndim != 4:
            raise ValueError(f"Expected image tensor [3,H,W] or [B,3,H,W], got {image.shape}")
        image = image.float()
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    return image.to(device)


def assign_scale_bin(log_area: float, scale_bin_edges: np.ndarray) -> int:
    s = np.searchsorted(scale_bin_edges, log_area, side="right") - 1
    s = np.clip(s, 0, len(scale_bin_edges) - 2)
    return int(s)


class PlacementScorer:
    def __init__(
        self,
        checkpoint_path: Path,
        preprocess_dir: Path,
        device: str = "cpu",
        img_size: int = 512,
    ):
        self.device = torch.device(device)
        self.img_size = img_size

        preprocess_dir = Path(preprocess_dir)
        checkpoint_path = Path(checkpoint_path)

        with open(preprocess_dir / "preprocess_config.json", "r") as f:
            config = json.load(f)

        self.grid_size = int(config["grid_size"])
        self.num_scales = int(config["num_scales"])
        self.scale_bin_edges = np.asarray(config["scale_bin_edges"], dtype=np.float64)

        backbone = MultiScaleBackbone()

        transformer = SimpleTransformer(
            embed_dim=1024,
            num_heads=8,
            num_layers=2,
            max_seq_len=self.grid_size * self.grid_size,
            pool=None,
        )

        decoder = Decoder(
            input_dim=1024,
            output_dim=self.num_scales,
        )

        self.model = MainModel(
            backbone=backbone,
            transformer=transformer,
            decoder=decoder,
        ).to(self.device)

        self.text_encoder = TextEncoder().to(self.device)
        self.text_encoder.eval()

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict_heatmap(self, image, class_name: str) -> np.ndarray:
        image = preprocess_image(image, self.device, img_size=self.img_size)

        with torch.no_grad():
            class_embeds = self.text_encoder([class_name])
            logits = self.model(image, class_embeds)

            if logits.ndim != 4:
                raise ValueError(f"Expected logits [B,S,H,W], got {logits.shape}")

            B, S, H, W = logits.shape

            probs = F.softmax(logits.reshape(B, -1), dim=1).reshape(B, S, H, W)

        return probs[0].detach().cpu().numpy()

    def score_bbox(self, image, class_name: str, bbox) -> float:
        heatmap = self.predict_heatmap(image, class_name)

        return bilinear_log_likelihood(
            heatmap=heatmap,
            bbox=bbox,
            scale_bin_edges=self.scale_bin_edges,
            grid_size=self.grid_size,
            num_scales=self.num_scales,
        )


def bilinear_log_likelihood(
    heatmap: np.ndarray,
    bbox,
    scale_bin_edges: np.ndarray,
    grid_size: int,
    num_scales: int,
) -> float:
    heatmap = np.asarray(heatmap, dtype=np.float64)

    if heatmap.shape != (num_scales, grid_size, grid_size):
        raise ValueError(
            f"Expected heatmap shape {(num_scales, grid_size, grid_size)}, got {heatmap.shape}"
        )

    x, y, w, h = [float(v) for v in bbox]

    if w <= 0 or h <= 0:
        return float("-inf")

    cx = x + w / 2
    cy = y + h / 2

    if not (0 <= cx <= 1 and 0 <= cy <= 1):
        return float("-inf")

    log_area = np.log(max(w * h, 1e-12))
    s_bin = assign_scale_bin(log_area, scale_bin_edges)

    gx = cx * (grid_size - 1)
    gy = cy * (grid_size - 1)

    gx0 = int(np.floor(gx))
    gy0 = int(np.floor(gy))
    gx1 = min(gx0 + 1, grid_size - 1)
    gy1 = min(gy0 + 1, grid_size - 1)

    fx = gx - gx0
    fy = gy - gy0

    p = (
        heatmap[s_bin, gy0, gx0] * (1 - fx) * (1 - fy)
        + heatmap[s_bin, gy0, gx1] * fx * (1 - fy)
        + heatmap[s_bin, gy1, gx0] * (1 - fx) * fy
        + heatmap[s_bin, gy1, gx1] * fx * fy
    )

    return float(np.log(max(float(p), 1e-12)))