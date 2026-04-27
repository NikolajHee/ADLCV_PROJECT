"""Evaluate Part C: score in-distribution and OOC test sets, compute AUROC."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve
from tqdm import tqdm

from src.config import (
    CHECKPOINTS_DIR,
    PLACES365_ROOT,
    PREPROCESSED_DIR,
)
from src.data.images import load_scene_image
from src.partC.inference import PlacementScorer


def score_test_set(
    scorer: PlacementScorer,
    examples: list[dict],
    places365_root: Path = PLACES365_ROOT,
) -> list[dict]:
    """Score every example in a test set.

    Returns:
        List of dicts with the original example fields plus 'log_likelihood'.
    """
    results = []

    for ex in tqdm(examples, desc="Scoring"):
        try:
            img = load_scene_image(ex["bg_path"], root=places365_root)
            log_lik = scorer.score_bbox(img, ex["fg_class"], ex["bbox"])
        except FileNotFoundError:
            log_lik = float("nan")

        results.append({**ex, "log_likelihood": log_lik})

    return results


def compute_auroc(
    in_dist_results: list[dict],
    ooc_results: list[dict],
) -> dict:
    """Compute AUROC and PR-AUC on combined results.

    Returns:
        Dict with 'auroc', 'pr_auc', 'fpr', 'tpr', 'precision', 'recall',
        'in_dist_scores', 'ooc_scores', plus counts.
    """
    in_scores = np.array(
        [r["log_likelihood"] for r in in_dist_results
         if not np.isnan(r["log_likelihood"]) and not np.isinf(r["log_likelihood"])]
    )
    ooc_scores = np.array(
        [r["log_likelihood"] for r in ooc_results
         if not np.isnan(r["log_likelihood"]) and not np.isinf(r["log_likelihood"])]
    )

    scores = np.concatenate([in_scores, ooc_scores])
    labels = np.concatenate([np.zeros(len(in_scores)), np.ones(len(ooc_scores))])

    auroc = roc_auc_score(labels, -scores)
    fpr, tpr, _ = roc_curve(labels, -scores)

    precision, recall, _ = precision_recall_curve(labels, -scores)
    pr_auc = auc(recall, precision)

    return {
        "auroc": float(auroc),
        "pr_auc": float(pr_auc),
        "n_in_dist": len(in_scores),
        "n_ooc": len(ooc_scores),
        "in_dist_scores": in_scores.tolist(),
        "ooc_scores": ooc_scores.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
    }


def evaluate(
    checkpoint_path: Path,
    in_dist_path: Path,
    ooc_path: Path,
    output_path: Path,
    device: str = "cpu",
) -> dict:
    """Full evaluation pipeline: load model, score test sets, compute metrics."""
    print(f"Loading scorer from {checkpoint_path}...")
    scorer = PlacementScorer(checkpoint_path=checkpoint_path, device=device)

    print(f"\nLoading test sets...")
    with open(in_dist_path) as f:
        in_dist_examples = json.load(f)
    with open(ooc_path) as f:
        ooc_examples = json.load(f)
    print(f"  In-distribution: {len(in_dist_examples)} examples")
    print(f"  OOC:             {len(ooc_examples)} examples")

    print("\nScoring in-distribution set...")
    in_dist_results = score_test_set(scorer, in_dist_examples)

    print("\nScoring OOC set...")
    ooc_results = score_test_set(scorer, ooc_examples)

    print("\nComputing metrics...")
    metrics = compute_auroc(in_dist_results, ooc_results)

    print(f"\n=== Results ===")
    print(f"AUROC:  {metrics['auroc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"In-distribution scores: mean={np.mean(metrics['in_dist_scores']):.3f}, "
          f"std={np.std(metrics['in_dist_scores']):.3f}")
    print(f"OOC scores:             mean={np.mean(metrics['ooc_scores']):.3f}, "
          f"std={np.std(metrics['ooc_scores']):.3f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "in_dist_path": str(in_dist_path),
                "ooc_path": str(ooc_path),
                "metrics": metrics,
                "in_dist_results": in_dist_results,
                "ooc_results": ooc_results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved results to {output_path}")

    return metrics


if __name__ == "__main__":
    ckpt_path = CHECKPOINTS_DIR / "smoke_test_best.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        raise SystemExit(1)

    in_dist_path = PREPROCESSED_DIR / "partC_test_sets" / "in_distribution.json"
    ooc_path = PREPROCESSED_DIR / "partC_test_sets" / "ooc_class_swap.json"
    output_path = CHECKPOINTS_DIR / "partC_results_smoke_test.json"

    evaluate(
        checkpoint_path=ckpt_path,
        in_dist_path=in_dist_path,
        ooc_path=ooc_path,
        output_path=output_path,
        device="cpu",  # smoke test, CPU is fine
    )