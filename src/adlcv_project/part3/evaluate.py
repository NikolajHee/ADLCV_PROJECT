"""Evaluate Part C: score in-distribution and OOC test sets, compute AUROC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve
from tqdm import tqdm

from adlcv_project.part3.inference import PlacementScorer


def load_scene_image(bg_path: str | Path, root: Path | None = None) -> Image.Image:
    bg_path = Path(bg_path)

    if bg_path.is_absolute():
        image_path = bg_path
    elif root is not None:
        image_path = root / bg_path
    else:
        image_path = bg_path

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    return Image.open(image_path).convert("RGB")


def load_json_list(path: Path) -> list[dict]:
    with open(path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected {path} to contain a list.")

    return data


def score_test_set(
    scorer: PlacementScorer,
    examples: list[dict],
    image_root: Path | None = None,
) -> list[dict]:
    results = []

    for ex in tqdm(examples, desc="Scoring"):
        try:
            img = load_scene_image(ex["bg_path"], root=image_root)

            log_lik = scorer.score_bbox(
                image=img,
                class_name=ex["fg_class"],
                bbox=ex["bbox"],
            )

            log_lik = float(log_lik)

        except FileNotFoundError as e:
            print(f"Missing image: {e}")
            log_lik = float("nan")

        except Exception as e:
            print(f"Failed example {ex.get('entry_id', 'unknown')}: {e}")
            log_lik = float("nan")

        results.append({**ex, "log_likelihood": log_lik})

    return results


def valid_scores(results: list[dict]) -> np.ndarray:
    scores = np.asarray([r["log_likelihood"] for r in results], dtype=float)
    return scores[np.isfinite(scores)]


def compute_auroc(in_dist_results: list[dict], ooc_results: list[dict]) -> dict:
    in_scores = valid_scores(in_dist_results)
    ooc_scores = valid_scores(ooc_results)

    if len(in_scores) == 0:
        raise ValueError("No valid in-distribution scores.")
    if len(ooc_scores) == 0:
        raise ValueError("No valid OOC scores.")

    scores = np.concatenate([in_scores, ooc_scores])

    labels = np.concatenate([
        np.zeros(len(in_scores), dtype=int),
        np.ones(len(ooc_scores), dtype=int),
    ])

    anomaly_scores = -scores

    auroc = roc_auc_score(labels, anomaly_scores)
    fpr, tpr, _ = roc_curve(labels, anomaly_scores)

    precision, recall, _ = precision_recall_curve(labels, anomaly_scores)
    pr_auc = auc(recall, precision)

    return {
        "auroc": float(auroc),
        "pr_auc": float(pr_auc),
        "n_in_dist": int(len(in_scores)),
        "n_ooc": int(len(ooc_scores)),
        "in_dist_scores": in_scores.tolist(),
        "ooc_scores": ooc_scores.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
    }


def evaluate(
    checkpoint_path: Path,
    preprocess_dir: Path,
    in_dist_path: Path,
    ooc_path: Path,
    output_path: Path,
    image_root: Path | None = None,
    device: str = "cpu",
) -> dict:
    print(f"Loading scorer from {checkpoint_path}")

    scorer = PlacementScorer(
        checkpoint_path=checkpoint_path,
        preprocess_dir=preprocess_dir,
        device=device,
    )

    print("\nLoading test sets")
    in_dist_examples = load_json_list(in_dist_path)
    ooc_examples = load_json_list(ooc_path)

    print(f"  In-distribution: {len(in_dist_examples)} examples")
    print(f"  OOC:             {len(ooc_examples)} examples")

    print("\nScoring in-distribution set")
    in_dist_results = score_test_set(
        scorer=scorer,
        examples=in_dist_examples,
        image_root=image_root,
    )

    print("\nScoring OOC set")
    ooc_results = score_test_set(
        scorer=scorer,
        examples=ooc_examples,
        image_root=image_root,
    )

    print("\nComputing metrics")
    metrics = compute_auroc(in_dist_results, ooc_results)

    print("\n=== Results ===")
    print(f"AUROC:  {metrics['auroc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"In-distribution mean: {np.mean(metrics['in_dist_scores']):.3f}")
    print(f"OOC mean:             {np.mean(metrics['ooc_scores']):.3f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "preprocess_dir": str(preprocess_dir),
                "in_dist_path": str(in_dist_path),
                "ooc_path": str(ooc_path),
                "image_root": str(image_root) if image_root is not None else None,
                "metrics": metrics,
                "in_dist_results": in_dist_results,
                "ooc_results": ooc_results,
            },
            f,
            indent=2,
        )

    print(f"\nSaved results to {output_path}")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--preprocess-dir", type=Path, default=Path("data/preprocessed_targets_top20"))
    parser.add_argument("--in-dist", type=Path, required=True)
    parser.add_argument("--ooc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("data"))
    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        preprocess_dir=args.preprocess_dir,
        in_dist_path=args.in_dist,
        ooc_path=args.ooc,
        output_path=args.output,
        image_root=args.image_root,
        device=args.device,
    )