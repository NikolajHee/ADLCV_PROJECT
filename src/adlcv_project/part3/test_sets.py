"""Build in-distribution and OOC test sets for Part C evaluation.

In-distribution set: high-reward positives from Marco's test split.
OOC set: same scenes/bboxes, but class swapped to something incongruous.

The OOC construction tests scene-class compatibility: the bbox itself is in a
plausible location for *its original class*, but the model is asked about a
different, mismatched class. A model that uses class info correctly should
assign lower probability to the OOC variant.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PREPROCESSED_DIR


# Class groupings for the OOC swap. We swap a class with something from a
# DIFFERENT group, so the swap is meaningfully incongruous (not just "swap
# orange for apple"). Groups are coarse — kitchen-y vs outdoor-vehicle vs etc.
CLASS_GROUPS: dict[str, list[str]] = {
    "kitchen": ["bottle", "bowl", "cup", "fork", "knife", "spoon", "wine glass",
                "pizza", "cake", "donut", "sandwich", "apple", "banana",
                "orange", "broccoli", "carrot", "hot dog"],
    "vehicle_outdoor": ["airplane", "boat", "bus", "car", "truck", "train",
                        "motorcycle", "bicycle"],
    "indoor_furniture": ["bed", "chair", "couch", "dining table", "tv",
                         "laptop", "keyboard", "mouse", "remote", "book",
                         "clock", "vase", "scissors"],
    "animal": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
               "bear", "zebra", "giraffe"],
    "outdoor_sports": ["surfboard", "skateboard", "skis", "snowboard",
                       "tennis racket", "baseball bat", "baseball glove",
                       "frisbee", "kite", "sports ball", "bench", "backpack"],
}


def _build_class_to_group(groups: dict[str, list[str]]) -> dict[str, str]:
    return {cls: group for group, classes in groups.items() for cls in classes}


def _pick_incongruous_class(original: str, all_classes: list[str], rng: random.Random) -> str:
    """Pick a class from a different group than `original`."""
    class_to_group = _build_class_to_group(CLASS_GROUPS)
    original_group = class_to_group.get(original)

    if original_group is None:
        return rng.choice([c for c in all_classes if c != original])

    candidates = [
        c for c in all_classes
        if class_to_group.get(c) is not None and class_to_group[c] != original_group
    ]
    return rng.choice(candidates) if candidates else rng.choice(all_classes)


def build_in_distribution_set(
    df_test: pd.DataFrame,
    n_samples: int,
    min_reward: float = -0.5,
    seed: int = 0,
) -> list[dict]:
    """Sample high-reward positives from the test split.

    Args:
        df_test: DataFrame of test annotations.
        n_samples: how many examples to sample.
        min_reward: only keep rows with image_reward_score >= this value.
        seed: random seed.

    Returns:
        List of dicts with keys: bg_path, fg_class, bbox, image_reward_score.
    """
    eligible = df_test[
        (df_test["label"] == 1)
        & (df_test["image_reward_score"] >= min_reward)
    ]
    print(f"Eligible in-distribution rows: {len(eligible):,}")

    sample = eligible.sample(n=min(n_samples, len(eligible)), random_state=seed)

    return [
        {
            "bg_path": row["bg_path"],
            "fg_class": row["fg_class"],
            "bbox": [float(x) for x in row["bbox"]],
            "image_reward_score": float(row["image_reward_score"]),
            "is_anomalous": False,
        }
        for _, row in sample.iterrows()
    ]


def build_ooc_set_class_swap(
    in_dist_set: list[dict],
    all_classes: list[str],
    seed: int = 0,
) -> list[dict]:
    """For each in-distribution example, build an OOC variant with class swapped."""
    rng = random.Random(seed)

    return [
        {
            "bg_path": ex["bg_path"],
            "fg_class": _pick_incongruous_class(ex["fg_class"], all_classes, rng),
            "original_class": ex["fg_class"],
            "bbox": [float(x) for x in ex["bbox"]],
            "image_reward_score": ex["image_reward_score"],
            "is_anomalous": True,
        }
        for ex in in_dist_set
    ]


def save_test_set(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(examples, f, indent=2)


def load_test_set(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    from src.data.annotations import load_as_dataframe

    out_dir = PREPROCESSED_DIR / "partC_test_sets"

    print("Loading test split...")
    df_test = load_as_dataframe("test")
    print(f"Test rows: {len(df_test):,}")

    with open(PREPROCESSED_DIR / "class_to_id.json") as f:
        all_classes = sorted(json.load(f).keys())

    in_dist = build_in_distribution_set(df_test, n_samples=500, min_reward=-0.5, seed=0)
    print(f"\nIn-distribution test set: {len(in_dist)} examples")
    print(f"  example: {in_dist[0]}")

    ooc = build_ooc_set_class_swap(in_dist, all_classes=all_classes, seed=0)
    print(f"\nOOC test set: {len(ooc)} examples")
    print(f"  example (anomalous):")
    print(f"    bg_path:        {ooc[0]['bg_path']}")
    print(f"    original_class: {ooc[0]['original_class']}")
    print(f"    swapped to:     {ooc[0]['fg_class']}")
    print(f"    bbox:           {ooc[0]['bbox']}")

    save_test_set(in_dist, out_dir / "in_distribution.json")
    save_test_set(ooc, out_dir / "ooc_class_swap.json")
    print(f"\nSaved to {out_dir}/")