import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def softmax_np(x, temperature=1.0):
    x = np.asarray(x, dtype=np.float64)
    temperature = max(float(temperature), 1e-8)
    x = x / temperature
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / (np.sum(exp_x) + 1e-12)


def bbox_xywh_to_center_scale(bbox):
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0
    area = max(w * h, 1e-12)
    log_area = np.log(area)
    return cx, cy, w, h, area, log_area


def assign_scale_bin(log_area, scale_bin_edges):
    s = np.searchsorted(scale_bin_edges, log_area, side="right") - 1
    s = np.clip(s, 0, len(scale_bin_edges) - 2)
    return int(s)


def compute_global_scale_bin_edges(hf_data, label=1, num_scales=8):
    log_areas = []

    for row in tqdm(hf_data, desc="Computing scale bins"):
        if row["label"] != label:
            continue
        _, _, _, _, _, log_area = bbox_xywh_to_center_scale(row["bbox"])
        log_areas.append(log_area)

    log_areas = np.asarray(log_areas, dtype=np.float64)
    if len(log_areas) == 0:
        raise ValueError("No valid rows found for scale bins.")

    min_la = np.min(log_areas)
    max_la = np.max(log_areas)

    if np.isclose(min_la, max_la):
        max_la = min_la + 1e-3

    return np.linspace(min_la, max_la, num_scales + 1)


def build_class_vocab(hf_data):
    classes = sorted({row["fg_class"] for row in hf_data})
    class_to_id = {cls_name: i for i, cls_name in enumerate(classes)}
    id_to_class = {i: cls_name for cls_name, i in class_to_id.items()}
    return class_to_id, id_to_class


def group_rows_by_scene_and_class(hf_data, label=1):
    grouped = defaultdict(list)

    for row in tqdm(hf_data, desc="Grouping rows"):
        if row["label"] != label:
            continue
        key = (row["bg_path"], row["fg_class"])
        grouped[key].append(row)

    return grouped

def filter_top_k_rows(
    rows,
    top_k=None,
    score_key="image_reward_score",
    reward_higher_is_better=True,
):
    if top_k is None or len(rows) <= top_k:
        return rows

    return sorted(
        rows,
        key=lambda r: r[score_key],
        reverse=reward_higher_is_better,
    )[:top_k]


def split_grouped_by_bg_path(grouped, val_fraction=0.1, seed=42):
    rng = np.random.default_rng(seed)

    bg_paths = sorted({bg_path for bg_path, fg_class in grouped.keys()})
    rng.shuffle(bg_paths)

    num_val = int(len(bg_paths) * val_fraction)
    val_bg_paths = set(bg_paths[:num_val])

    train_grouped = {}
    val_grouped = {}

    for key, rows in grouped.items():
        bg_path, fg_class = key

        if bg_path in val_bg_paths:
            val_grouped[key] = rows
        else:
            train_grouped[key] = rows

    return train_grouped, val_grouped

def build_aggregate_target_3d(
    rows,
    grid_size=32,
    num_scales=8,
    sigma_xy=1.25,
    sigma_s=0.6,
    score_temperature=0.3,
    reward_higher_is_better=False,
    score_key="image_reward_score",
    scale_bin_edges=None,
    normalize=True,
    dtype=np.float32,
):
    if len(rows) == 0:
        raise ValueError("rows must be non-empty")

    if scale_bin_edges is None:
        raise ValueError("scale_bin_edges should be provided globally for full preprocessing.")

    raw_scores = np.asarray([row[score_key] for row in rows], dtype=np.float64)

    if reward_higher_is_better:
        score_for_weight = raw_scores
    else:
        score_for_weight = -raw_scores

    weights = softmax_np(score_for_weight, temperature=score_temperature)

    target = np.zeros((num_scales, grid_size, grid_size), dtype=np.float64)

    xs = np.arange(grid_size, dtype=np.float64)[None, None, :]
    ys = np.arange(grid_size, dtype=np.float64)[None, :, None]
    ss = np.arange(num_scales, dtype=np.float64)[:, None, None]

    for row, weight in zip(rows, weights):
        cx, cy, w, h, area, log_area = bbox_xywh_to_center_scale(row["bbox"])

        gx = cx * (grid_size - 1)
        gy = cy * (grid_size - 1)
        s_bin = assign_scale_bin(log_area, scale_bin_edges)

        loc_kernel = np.exp(
            -((xs - gx) ** 2 + (ys - gy) ** 2) / (2.0 * sigma_xy ** 2)
        )
        scale_kernel = np.exp(
            -((ss - s_bin) ** 2) / (2.0 * sigma_s ** 2)
        )

        target += weight * scale_kernel * loc_kernel

    if normalize:
        target_sum = target.sum()
        if target_sum > 0:
            target /= target_sum

    return target.astype(dtype), {
        "weights": weights,
        "raw_scores": raw_scores,
        "scale_bin_edges": scale_bin_edges,
    }

def save_grouped_targets(
    grouped,
    output_dir,
    class_to_id,
    scale_bin_edges,
    label=1,
    grid_size=32,
    num_scales=8,
    sigma_xy=1.25,
    sigma_s=0.6,
    score_temperature=0.3,
    reward_higher_is_better=True,
    top_k=20,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index = []

    for sample_id, ((bg_path, fg_class), rows) in enumerate(
        tqdm(grouped.items(), desc=f"Saving targets to {output_dir}")
    ):
        rows_used = filter_top_k_rows(
            rows,
            top_k=top_k,
            score_key="image_reward_score",
            reward_higher_is_better=reward_higher_is_better,
        )

        class_id = class_to_id[fg_class]

        target, meta = build_aggregate_target_3d(
            rows=rows_used,
            grid_size=grid_size,
            num_scales=num_scales,
            sigma_xy=sigma_xy,
            sigma_s=sigma_s,
            score_temperature=score_temperature,
            reward_higher_is_better=reward_higher_is_better,
            scale_bin_edges=scale_bin_edges,
        )

        save_path = output_dir / f"{sample_id:07d}.npz"

        np.savez_compressed(
            save_path,
            target=target,
            bg_path=np.array(bg_path),
            fg_class=np.array(fg_class),
            class_id=np.int64(class_id),
            num_rows=np.int64(len(rows)),
            num_rows_used=np.int64(len(rows_used)),
        )

        index.append({
            "sample_id": sample_id,
            "bg_path": bg_path,
            "fg_class": fg_class,
            "class_id": class_id,
            "target_path": str(save_path),
            "num_rows": len(rows),
            "num_rows_used": len(rows_used),
        })

    with open(output_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    return index

def preprocess_training_targets(
    dataset,
    output_dir,
    label=1,
    grid_size=32,
    num_scales=8,
    sigma_xy=1.25,
    sigma_s=0.6,
    score_temperature=0.3,
    reward_higher_is_better=True,
    top_k=20,
    val_fraction=0.1,
    seed=42,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_data = dataset.hf_data

    class_to_id, id_to_class = build_class_vocab(hf_data)

    scale_bin_edges = compute_global_scale_bin_edges(
        hf_data,
        label=label,
        num_scales=num_scales,
    )

    grouped = group_rows_by_scene_and_class(hf_data, label=label)

    train_grouped, val_grouped = split_grouped_by_bg_path(
        grouped,
        val_fraction=val_fraction,
        seed=seed,
    )

    train_index = save_grouped_targets(
        grouped=train_grouped,
        output_dir=output_dir / "train",
        class_to_id=class_to_id,
        scale_bin_edges=scale_bin_edges,
        label=label,
        grid_size=grid_size,
        num_scales=num_scales,
        sigma_xy=sigma_xy,
        sigma_s=sigma_s,
        score_temperature=score_temperature,
        reward_higher_is_better=reward_higher_is_better,
        top_k=top_k,
    )

    val_index = save_grouped_targets(
        grouped=val_grouped,
        output_dir=output_dir / "val",
        class_to_id=class_to_id,
        scale_bin_edges=scale_bin_edges,
        label=label,
        grid_size=grid_size,
        num_scales=num_scales,
        sigma_xy=sigma_xy,
        sigma_s=sigma_s,
        score_temperature=score_temperature,
        reward_higher_is_better=reward_higher_is_better,
        top_k=top_k,
    )

    with open(output_dir / "class_to_id.json", "w", encoding="utf-8") as f:
        json.dump(class_to_id, f, indent=2)

    with open(output_dir / "preprocess_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "label": label,
            "grid_size": grid_size,
            "num_scales": num_scales,
            "sigma_xy": sigma_xy,
            "sigma_s": sigma_s,
            "score_temperature": score_temperature,
            "reward_higher_is_better": reward_higher_is_better,
            "top_k": top_k,
            "val_fraction": val_fraction,
            "seed": seed,
            "scale_bin_edges": scale_bin_edges.tolist(),
            "num_train_samples": len(train_index),
            "num_val_samples": len(val_index),
        }, f, indent=2)

    return train_index, val_index, class_to_id, scale_bin_edges



def main():
    from adlcv_project.data import HiddenObjectsDatasetStreaming

    dataset = HiddenObjectsDatasetStreaming("data", split="train")

    preprocess_training_targets(
        dataset,
        output_dir="data/preprocessed_targets_top20",
        label=1,
        grid_size=32,
        num_scales=8,
        sigma_xy=1.25,
        sigma_s=0.6,
        score_temperature=0.3,
        reward_higher_is_better=True,
        top_k=20,
        val_fraction=0.1,
        seed=42,
    )

if __name__ == "__main__":
    main()