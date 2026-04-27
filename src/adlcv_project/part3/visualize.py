"""Visualization for Part C evaluation results.

Two main figures:
- score_distribution_and_curves: histograms + ROC + PR curves (the headline figure)
- qualitative_gallery: image grid with predicted heatmaps and likelihood scores
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

from src.config import CHECKPOINTS_DIR, FIGURES_DIR, PLACES365_ROOT
from src.data.images import load_scene_image
from src.partC.inference import PlacementScorer


def plot_score_distribution_and_curves(
    metrics: dict,
    output_path: Path | None = None,
    title_suffix: str = "",
) -> None:
    """Three-panel figure: score histograms, ROC curve, PR curve."""
    in_scores = np.array(metrics["in_dist_scores"])
    ooc_scores = np.array(metrics["ooc_scores"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    bins = np.linspace(
        min(in_scores.min(), ooc_scores.min()),
        max(in_scores.max(), ooc_scores.max()),
        50,
    )
    axes[0].hist(in_scores, bins=bins, alpha=0.6, label=f"In-distribution (n={len(in_scores)})", color="tab:green")
    axes[0].hist(ooc_scores, bins=bins, alpha=0.6, label=f"OOC (n={len(ooc_scores)})", color="tab:red")
    axes[0].axvline(in_scores.mean(), color="tab:green", linestyle="--", alpha=0.7, label=f"in-dist mean = {in_scores.mean():.2f}")
    axes[0].axvline(ooc_scores.mean(), color="tab:red", linestyle="--", alpha=0.7, label=f"OOC mean = {ooc_scores.mean():.2f}")
    axes[0].set_xlabel("Log-likelihood")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Score distributions{title_suffix}")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    fpr = np.array(metrics["fpr"])
    tpr = np.array(metrics["tpr"])
    auroc = metrics["auroc"]
    axes[1].plot(fpr, tpr, color="tab:blue", linewidth=2, label=f"AUROC = {auroc:.3f}")
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.5, label="random")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_title(f"ROC curve{title_suffix}")
    axes[1].legend(fontsize=9, loc="lower right")
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)

    precision = np.array(metrics["precision"])
    recall = np.array(metrics["recall"])
    pr_auc = metrics["pr_auc"]
    axes[2].plot(recall, precision, color="tab:purple", linewidth=2, label=f"PR-AUC = {pr_auc:.3f}")

    base_rate = len(ooc_scores) / (len(in_scores) + len(ooc_scores))
    axes[2].axhline(base_rate, color="gray", linestyle="--", alpha=0.5, label=f"random ({base_rate:.2f})")
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
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
    plt.show()


def plot_qualitative_gallery(
    results: list[dict],
    scorer: PlacementScorer,
    n_examples: int = 8,
    title: str = "",
    output_path: Path | None = None,
    seed: int = 0,
) -> None:
    """Sample examples from results, show image + predicted heatmap + bbox + score."""
    rng = np.random.default_rng(seed)
    valid = [r for r in results if not (np.isnan(r["log_likelihood"]) or np.isinf(r["log_likelihood"]))]

    sample = rng.choice(valid, size=min(n_examples, len(valid)), replace=False)

    n_cols = 4
    n_rows = (len(sample) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes)

    for i, ex in enumerate(sample):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]

        try:
            img = load_scene_image(ex["bg_path"])
        except FileNotFoundError:
            ax.text(0.5, 0.5, f"Missing image:\n{ex['bg_path']}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue

        heatmap = scorer.predict_heatmap(img, ex["fg_class"])
        heatmap_2d = heatmap.sum(axis=0)
        heatmap_up = zoom(heatmap_2d, 512 / 32, order=1)

        ax.imshow(img)
        ax.imshow(heatmap_up, cmap="hot", alpha=0.5)

        x, y, w, h = ex["bbox"]
        rect = plt.Rectangle(
            (x * 512, y * 512), w * 512, h * 512,
            fill=False, edgecolor="cyan", linewidth=2,
        )
        ax.add_patch(rect)

        is_anom = ex.get("is_anomalous", False)
        ll = ex["log_likelihood"]
        marker = "[OOC]" if is_anom else "[in-dist]"

        if is_anom and "original_class" in ex:
            title_str = f"{marker} class={ex['fg_class']} (was {ex['original_class']})\nlog-lik={ll:.2f}"
        else:
            title_str = f"{marker} class={ex['fg_class']}\nlog-lik={ll:.2f}"

        ax.set_title(title_str, fontsize=9)
        ax.axis("off")

    for j in range(len(sample), n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis("off")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
    plt.show()


if __name__ == "__main__":
    results_path = CHECKPOINTS_DIR / "partC_results_smoke_test.json"
    if not results_path.exists():
        print(f"Results not found: {results_path}")
        print("Run src.partC.evaluate first.")
        raise SystemExit(1)

    with open(results_path) as f:
        results_data = json.load(f)

    metrics = results_data["metrics"]
    title_suffix = " (smoke test)"

    plot_score_distribution_and_curves(
        metrics,
        output_path=FIGURES_DIR / "partC_eval_smoke_test.png",
        title_suffix=title_suffix,
    )

    ckpt_path = Path(results_data["checkpoint"])
    scorer = PlacementScorer(checkpoint_path=ckpt_path, device="cpu")

    plot_qualitative_gallery(
        results_data["in_dist_results"],
        scorer,
        n_examples=8,
        title=f"In-distribution gallery{title_suffix}",
        output_path=FIGURES_DIR / "partC_gallery_in_dist_smoke_test.png",
    )

    plot_qualitative_gallery(
        results_data["ooc_results"],
        scorer,
        n_examples=8,
        title=f"OOC gallery{title_suffix}",
        output_path=FIGURES_DIR / "partC_gallery_ooc_smoke_test.png",
    )