"""Build Part C in-distribution and OOC validation sets using our preprocessing split."""

from __future__ import annotations

import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm


CLASS_GROUPS = {
    "kitchen": [
        "bottle", "bowl", "cup", "fork", "knife", "spoon", "wine glass",
        "pizza", "cake", "donut", "sandwich", "apple", "banana",
        "orange", "broccoli", "carrot", "hot dog",
    ],
    "vehicle_outdoor": [
        "airplane", "boat", "bus", "car", "truck", "train",
        "motorcycle", "bicycle",
    ],
    "indoor_furniture": [
        "bed", "chair", "couch", "dining table", "tv",
        "laptop", "keyboard", "mouse", "remote", "book",
        "clock", "vase", "scissors",
    ],
    "animal": [
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
        "bear", "zebra", "giraffe",
    ],
    "outdoor_sports": [
        "surfboard", "skateboard", "skis", "snowboard",
        "tennis racket", "baseball bat", "baseball glove",
        "frisbee", "kite", "sports ball", "bench", "backpack",
    ],
}


def build_class_to_group():
    return {
        cls: group
        for group, classes in CLASS_GROUPS.items()
        for cls in classes
    }


def pick_incongruous_class(original_class, all_classes, rng):
    class_to_group = build_class_to_group()
    original_group = class_to_group.get(original_class)

    if original_group is None:
        candidates = [c for c in all_classes if c != original_class]
    else:
        candidates = [
            c for c in all_classes
            if c != original_class
            and c in class_to_group
            and class_to_group[c] != original_group
        ]

    if len(candidates) == 0:
        candidates = [c for c in all_classes if c != original_class]

    return rng.choice(candidates)


def group_rows_by_scene_and_class(hf_data, label=1):
    grouped = defaultdict(list)

    for row in tqdm(hf_data, desc="Grouping rows"):
        if row["label"] != label:
            continue

        key = (row["bg_path"], row["fg_class"])
        grouped[key].append(row)

    return grouped


def split_grouped_by_bg_path(grouped, val_fraction=0.1, seed=42):
    rng = np.random.default_rng(seed)

    bg_paths = sorted({bg_path for bg_path, _ in grouped.keys()})
    rng.shuffle(bg_paths)

    num_val = int(len(bg_paths) * val_fraction)
    val_bg_paths = set(bg_paths[:num_val])

    train_grouped = {}
    val_grouped = {}

    for key, rows in grouped.items():
        bg_path, _ = key

        if bg_path in val_bg_paths:
            val_grouped[key] = rows
        else:
            train_grouped[key] = rows

    return train_grouped, val_grouped


def select_best_row(rows, score_key="image_reward_score", reward_higher_is_better=True):
    return sorted(
        rows,
        key=lambda r: r[score_key],
        reverse=reward_higher_is_better,
    )[0]


def build_in_distribution_set(
    grouped,
    n_samples=500,
    min_reward=None,
    seed=0,
    reward_higher_is_better=True,
):
    rng = random.Random(seed)

    examples = []

    for (bg_path, fg_class), rows in grouped.items():
        if len(rows) == 0:
            continue

        best = select_best_row(
            rows,
            score_key="image_reward_score",
            reward_higher_is_better=reward_higher_is_better,
        )

        score = float(best["image_reward_score"])

        if min_reward is not None and score < min_reward:
            continue

        examples.append(
            {
                "entry_id": best.get("entry_id", None),
                "bg_path": bg_path,
                "fg_class": fg_class,
                "bbox": [float(x) for x in best["bbox"]],
                "image_reward_score": score,
                "label": 1,
                "is_anomalous": False,
                "source": best.get("source", None),
            }
        )

    rng.shuffle(examples)

    return examples[: min(n_samples, len(examples))]


def build_ooc_set_class_swap(in_dist_set, all_classes, seed=0):
    rng = random.Random(seed)

    ooc = []

    for ex in in_dist_set:
        swapped_class = pick_incongruous_class(
            original_class=ex["fg_class"],
            all_classes=all_classes,
            rng=rng,
        )

        ooc.append(
            {
                "entry_id": ex.get("entry_id", None),
                "bg_path": ex["bg_path"],
                "fg_class": swapped_class,
                "original_class": ex["fg_class"],
                "bbox": [float(x) for x in ex["bbox"]],
                "image_reward_score": float(ex["image_reward_score"]),
                "label": 0,
                "is_anomalous": True,
                "source": ex.get("source", None),
            }
        )

    return ooc


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    from adlcv_project.data import HiddenObjectsDatasetStreaming

    data_root = "data"
    output_dir = Path("data/preprocessed_targets_top20/partC_test_sets")

    label = 1
    val_fraction = 0.1
    split_seed = 42
    sample_seed = 0
    n_samples = 500

    dataset = HiddenObjectsDatasetStreaming(data_root, split="train")
    hf_data = dataset.hf_data

    print("Building class vocabulary...")
    all_classes = sorted({row["fg_class"] for row in hf_data})
    print(f"Number of classes: {len(all_classes)}")

    print("Grouping rows...")
    grouped = group_rows_by_scene_and_class(hf_data, label=label)

    print("Recreating same train/val split as preprocessing...")
    _, val_grouped = split_grouped_by_bg_path(
        grouped,
        val_fraction=val_fraction,
        seed=split_seed,
    )

    print(f"Validation scene/class groups: {len(val_grouped):,}")

    print("Building in-distribution set...")
    in_dist = build_in_distribution_set(
        val_grouped,
        n_samples=n_samples,
        min_reward=None,
        seed=sample_seed,
        reward_higher_is_better=True,
    )

    print("Building OOC class-swap set...")
    ooc = build_ooc_set_class_swap(
        in_dist,
        all_classes=all_classes,
        seed=sample_seed,
    )

    save_json(in_dist, output_dir / "in_distribution.json")
    save_json(ooc, output_dir / "ooc_class_swap.json")

    print(f"\nSaved to: {output_dir}")
    print(f"In-distribution examples: {len(in_dist)}")
    print(f"OOC examples:             {len(ooc)}")

    if len(in_dist) > 0:
        print("\nExample:")
        print("IN: ", in_dist[0])
        print("OOC:", ooc[0])


if __name__ == "__main__":
    main()