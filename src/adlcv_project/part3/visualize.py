"""Visualization for Part C evaluation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom

from adlcv_project.part3.inference import PlacementScorer


def load_scene_image(bg_path: str | Path, root: Path | None = None) -> Image.Image:
    bg_path = Path(bg_path)

    if bg_path.is_absolute():
        path = bg_path
    elif root is not None:
        path = root / bg_path
    else:
        path = bg_path

    if not path.exists():
        raise FileNotFoundError(path)

    return Image.open(path).convert("RGB")


def to_numpy_heatmap(heatmap):
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()

    heatmap = np.asarray(heatmap)

    if heatmap.ndim == 4:
        heatmap = heatmap[0]

    return heatmap


def plot_score_distribution_and_curves(
    metrics: dict,
    output_path: Path | None = None,
    title_suffix: str = "",
) -> None:
    in_scores = np.asarray(metrics["in_dist_scores"], dtype=float)
    ooc_scores = np.asarray(metrics["ooc_scores"], dtype=float)

    in_scores = in_scores[np.isfinite(in_scores)]
    ooc_scores = ooc_scores[np.isfinite(ooc_scores)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    bins = np.linspace(
        min(in_scores.min(), ooc_scores.min()),
        max(in_scores.max(), ooc_scores.max()),
        50,
    )

    axes[0].hist(in_scores, bins=bins, alpha=0.6, label=f"In-dist n={len(in_scores)}")
    axes[0].hist(ooc_scores, bins=bins, alpha=0.6, label=f"OOC n={len(ooc_scores)}")
    axes[0].axvline(in_scores.mean(), linestyle="--", label=f"in mean={in_scores.mean():.2f}")
    axes[0].axvline(ooc_scores.mean(), linestyle="--", label=f"OOC mean={ooc_scores.mean():.2f}")
    axes[0].set_xlabel("Log-likelihood")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Score distributions{title_suffix}")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    fpr = np.asarray(metrics["fpr"], dtype=float)
    tpr = np.asarray(metrics["tpr"], dtype=float)

    axes[1].plot(fpr, tpr, linewidth=2, label=f"AUROC={metrics['auroc']:.3f}")
    axes[1].plot([0, 1], [0, 1], linestyle="--", alpha=0.5, label="random")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_title(f"ROC curve{title_suffix}")
    axes[1].legend(fontsize=9, loc="lower right")
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)

    precision = np.asarray(metrics["precision"], dtype=float)
    recall = np.asarray(metrics["recall"], dtype=float)

    axes[2].plot(recall, precision, linewidth=2, label=f"PR-AUC={metrics['pr_auc']:.3f}")

    base_rate = len(ooc_scores) / (len(in_scores) + len(ooc_scores))
    axes[2].axhline(base_rate, linestyle="--", alpha=0.5, label=f"random={base_rate:.2f}")
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")
    axes[2].set_title(f"Precision-Recall curve{title_suffix}")
    axes[2].legend(fontsize=9, loc="lower left")
    axes[2].grid(alpha=0.3)
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1.02)

    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    plt.show()


def plot_qualitative_gallery(
    results: list[dict],
    scorer: PlacementScorer,
    image_root: Path | None = None,
    n_examples: int = 8,
    title: str = "",
    output_path: Path | None = None,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)

    valid = [
        r for r in results
        if np.isfinite(float(r["log_likelihood"]))
    ]

    if len(valid) == 0:
        raise ValueError("No valid examples to plot.")

    sample = rng.choice(valid, size=min(n_examples, len(valid)), replace=False)

    n_cols = 4
    n_rows = int(np.ceil(len(sample) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.asarray(axes).reshape(n_rows, n_cols)

    for i, ex in enumerate(sample):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]

        try:
            img = load_scene_image(ex["bg_path"], root=image_root)
        except FileNotFoundError:
            ax.text(
                0.5,
                0.5,
                f"Missing image:\n{ex['bg_path']}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")
            continue

        img_w, img_h = img.size

        heatmap = scorer.predict_heatmap(img, ex["fg_class"])
        heatmap = to_numpy_heatmap(heatmap)

        if heatmap.ndim == 3:
            heatmap_2d = heatmap.sum(axis=0)
        elif heatmap.ndim == 2:
            heatmap_2d = heatmap
        else:
            raise ValueError(f"Unexpected heatmap shape: {heatmap.shape}")

        heatmap_h, heatmap_w = heatmap_2d.shape
        heatmap_up = zoom(heatmap_2d, (img_h / heatmap_h, img_w / heatmap_w), order=1)

        ax.imshow(img)
        ax.imshow(heatmap_up, cmap="hot", alpha=0.5)

        x, y, w, h = [float(v) for v in ex["bbox"]]

        rect = plt.Rectangle(
            (x * img_w, y * img_h),
            w * img_w,
            h * img_h,
            fill=False,
            edgecolor="cyan",
            linewidth=2,
        )
        ax.add_patch(rect)

        is_anom = ex.get("is_anomalous", False)
        ll = float(ex["log_likelihood"])
        marker = "OOC" if is_anom else "in-dist"

        if is_anom and "original_class" in ex:
            title_str = f"{marker}: {ex['fg_class']}\nwas {ex['original_class']} | ll={ll:.2f}"
        else:
            title_str = f"{marker}: {ex['fg_class']}\nll={ll:.2f}"

        ax.set_title(title_str, fontsize=9)
        ax.axis("off")

    for j in range(len(sample), n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis("off")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--preprocess-dir", type=Path, default=Path("data/preprocessed_targets_top20"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures/partC"))
    parser.add_argument("--image-root", type=Path, default=Path("data"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-examples", type=int, default=8)

    args = parser.parse_args()

    with open(args.results, "r") as f:
        results_data = json.load(f)

    metrics = results_data["metrics"]
    checkpoint_path = Path(results_data["checkpoint"])

    plot_score_distribution_and_curves(
        metrics,
        output_path=args.figures_dir / "partC_score_distribution_and_curves.png",
    )

    scorer = PlacementScorer(
        checkpoint_path=checkpoint_path,
        preprocess_dir=args.preprocess_dir,
        device=args.device,
    )

    plot_qualitative_gallery(
        results_data["in_dist_results"],
        scorer=scorer,
        image_root=args.image_root,
        n_examples=args.n_examples,
        title="In-distribution examples",
        output_path=args.figures_dir / "partC_gallery_in_distribution.png",
    )

    plot_qualitative_gallery(
        results_data["ooc_results"],
        scorer=scorer,
        image_root=args.image_root,
        n_examples=args.n_examples,
        title="OOC class-swap examples",
        output_path=args.figures_dir / "partC_gallery_ooc.png",
    )


if __name__ == "__main__":
    main()