from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

from adlcv_project.part3.inference import PlacementScorer
from adlcv_project.part3.visualize import load_scene_image, to_numpy_heatmap


def select_top_pairs(in_dist_results, ooc_results, n_pairs=2):
    if len(in_dist_results) != len(ooc_results):
        raise ValueError(
            f"Length mismatch: in_dist={len(in_dist_results)}, "
            f"ooc={len(ooc_results)}"
        )

    candidates = []

    for in_ex, ooc_ex in zip(in_dist_results, ooc_results):
        ll_in = float(in_ex["log_likelihood"])
        ll_ooc = float(ooc_ex["log_likelihood"])

        if not np.isfinite(ll_in) or not np.isfinite(ll_ooc):
            continue

        if in_ex["bg_path"] != ooc_ex["bg_path"]:
            continue

        diff = ll_in - ll_ooc
        candidates.append((in_ex, ooc_ex, diff))

    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[:n_pairs]


def render_panel(ax, example, scorer, image_root):
    img = load_scene_image(example["bg_path"], root=image_root)
    img_w, img_h = img.size

    heatmap = scorer.predict_heatmap(img, example["fg_class"])
    heatmap = to_numpy_heatmap(heatmap)

    if heatmap.ndim == 3:
        heatmap_2d = heatmap.sum(axis=0)
    else:
        heatmap_2d = heatmap

    heatmap_h, heatmap_w = heatmap_2d.shape
    heatmap_up = zoom(
        heatmap_2d,
        (img_h / heatmap_h, img_w / heatmap_w),
        order=1,
    )

    ax.imshow(img)
    ax.imshow(heatmap_up, cmap="hot", alpha=0.5)

    x, y, w, h = [float(v) for v in example["bbox"]]

    rect = plt.Rectangle(
        (x * img_w, y * img_h),
        w * img_w,
        h * img_h,
        fill=False,
        edgecolor="cyan",
        linewidth=2.5,
    )
    ax.add_patch(rect)

    ll = float(example["log_likelihood"])

    if example.get("is_anomalous", False):
        title = (
            f"OOC: {example['fg_class']}\n"
            f"was: {example.get('original_class', '?')}\n"
            f"log-lik = {ll:.2f}"
        )
    else:
        title = (
            f"In-distribution: {example['fg_class']}\n"
            f"\n"
            f"log-lik = {ll:.2f}"
        )

    ax.set_title(title, fontsize=10)
    ax.axis("off")


def plot_top_ooc_pairs(
    results_path,
    checkpoint_path=None,
    preprocess_dir=Path("data/preprocessed_targets_top20"),
    image_root=Path("data"),
    output_path=Path("figures/partC/top_ooc_pairs.png"),
    n_pairs=2,
    device="cpu",
):
    with open(results_path, "r") as f:
        results_data = json.load(f)

    if checkpoint_path is None:
        checkpoint_path = Path(results_data["checkpoint"])
    else:
        checkpoint_path = Path(checkpoint_path)

    in_dist_results = results_data["in_dist_results"]
    ooc_results = results_data["ooc_results"]

    top_pairs = select_top_pairs(
        in_dist_results=in_dist_results,
        ooc_results=ooc_results,
        n_pairs=n_pairs,
    )

    scorer = PlacementScorer(
        checkpoint_path=checkpoint_path,
        preprocess_dir=preprocess_dir,
        device=device,
    )

    fig, axes = plt.subplots(
        1,
        2 * n_pairs,
        figsize=(4.2 * 2 * n_pairs, 4.8),
        constrained_layout=True,
    )

    axes = np.asarray(axes).reshape(-1)

    for i, (in_ex, ooc_ex, diff) in enumerate(top_pairs):
        render_panel(
            axes[2 * i],
            in_ex,
            scorer=scorer,
            image_root=image_root,
        )

        render_panel(
            axes[2 * i + 1],
            ooc_ex,
            scorer=scorer,
            image_root=image_root,
        )

        axes[2 * i].text(
            0.5,
            -0.08,
            f"Difference: {diff:.2f}",
            transform=axes[2 * i].transAxes,
            ha="center",
            va="top",
            fontsize=10,
        )

    fig.suptitle(
        "Top in-distribution vs OOC pairs by log-likelihood difference",
        fontsize=14,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to: {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/partC_results.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--preprocess-dir",
        type=Path,
        default=Path("data/preprocessed_targets"),
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("data"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/partC/top_ooc_pairs.png"),
    )
    parser.add_argument("--n-pairs", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    plot_top_ooc_pairs(
        results_path=args.results,
        checkpoint_path=args.checkpoint,
        preprocess_dir=args.preprocess_dir,
        image_root=args.image_root,
        output_path=args.output,
        n_pairs=args.n_pairs,
        device=args.device,
    )


if __name__ == "__main__":
    main()