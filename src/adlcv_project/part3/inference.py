"""Score a specific bbox placement against a trained Part B model.

The model produces a [num_scales, grid_size, grid_size] probability distribution
over (scale, y, x) for a given (image, class). We extract the probability at
the query bbox's location using bilinear interpolation in the spatial dims.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.config import (
    CLASS_EMBEDDINGS_PATH,
    GRID_SIZE,
    NUM_SCALES,
    PLACES365_ROOT,
    SCALE_BIN_EDGES_PATH,
)
from src.data.images import load_scene_image
from src.partA.scale_bins import load_edges
from src.partB.model import PlacementModel


class PlacementScorer:
    """Wraps a trained PlacementModel for OOC-style likelihood queries."""

    def __init__(
        self,
        checkpoint_path: Path,
        device: str = "cpu",
        scale_bin_edges_path: Path = SCALE_BIN_EDGES_PATH,
        class_embeddings_path: Path = CLASS_EMBEDDINGS_PATH,
    ) -> None:
        self.device = device

        self.model = PlacementModel().to(device)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.scale_bin_edges = load_edges(scale_bin_edges_path)
        self.class_embeddings: dict[str, torch.Tensor] = torch.load(
            class_embeddings_path, weights_only=False
        )

    def predict_heatmap(
        self,
        image: torch.Tensor | np.ndarray,
        class_name: str,
    ) -> np.ndarray:
        """Run forward pass and return the full [num_scales, H, W] probability heatmap.

        Args:
            image: (3, 512, 512) tensor in [0, 1], or (512, 512, 3) uint8 array.
            class_name: Object class name (must exist in class_embeddings).

        Returns:
            np.ndarray of shape (num_scales, grid_size, grid_size), sums to 1.
        """
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        image = image.unsqueeze(0).to(self.device)
        cls_emb = self.class_embeddings[class_name].unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(image, cls_emb)
            B = logits.size(0)
            probs = F.softmax(logits.reshape(B, -1), dim=1).reshape(B, NUM_SCALES, GRID_SIZE, GRID_SIZE)

        return probs[0].cpu().numpy()

    def score_bbox(
        self,
        image: torch.Tensor | np.ndarray,
        class_name: str,
        bbox: list[float] | np.ndarray,
    ) -> float:
        """Return the log-likelihood of one specific bbox placement.

        Args:
            image: see predict_heatmap.
            class_name: object class name.
            bbox: normalized [x, y, w, h] in [0, 1].

        Returns:
            log-likelihood (single float). Higher = more plausible.
        """
        heatmap = self.predict_heatmap(image, class_name)
        return _bilinear_log_likelihood(heatmap, bbox, self.scale_bin_edges)


def _bilinear_log_likelihood(
    heatmap: np.ndarray,
    bbox: list[float] | np.ndarray,
    scale_bin_edges: np.ndarray,
    grid_size: int = GRID_SIZE,
    num_scales: int = NUM_SCALES,
) -> float:
    """Look up the heatmap probability at the bbox's (scale, gy, gx) using
    bilinear interpolation in the spatial dims, then return its log."""
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0

    if not (0 <= cx <= 1 and 0 <= cy <= 1):
        return float("-inf")

    log_area = np.log(max(w * h, 1e-12))
    s_bin = int(np.clip(
        np.searchsorted(scale_bin_edges, log_area, side="right") - 1,
        0, num_scales - 1,
    ))

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

    return float(np.log(max(p, 1e-12)))


if __name__ == "__main__":
    from src.config import CHECKPOINTS_DIR

    ckpt_path = CHECKPOINTS_DIR / "smoke_test_best.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        print("Train at least the smoke test first, or point at a different checkpoint.")
        raise SystemExit(1)

    scorer = PlacementScorer(checkpoint_path=ckpt_path, device="cpu")

    img = load_scene_image("data_large_standard/k/kitchen/00002986.jpg")

    plausible_bbox = [0.55, 0.65, 0.10, 0.20]
    implausible_bbox = [0.10, 0.10, 0.10, 0.20]

    score_a = scorer.score_bbox(img, "bottle", plausible_bbox)
    score_b = scorer.score_bbox(img, "bottle", implausible_bbox)

    print(f"Bottle in plausible spot (counter):     log-lik = {score_a:.3f}")
    print(f"Bottle in implausible spot (top-left):  log-lik = {score_b:.3f}")
    print(f"\nPlausible should score higher than implausible.")
    print(f"With smoke-test model (barely trained), this may not hold reliably.")