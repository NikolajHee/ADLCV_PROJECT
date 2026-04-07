import os
import shutil
from datasets import load_dataset
from tqdm import tqdm

# --- SETTINGS ---
original_places_root = "D:/places365"
subset_places_root = "./data/places365_subset"
splits = ("train", "test")   # hidden-objects uses train/test
use_symlinks = False         # True saves disk space, but may need Windows Developer Mode/admin

# Optional: silence the HF symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# --- QUICK PATH CHECK ---
ds_check = load_dataset("marco-schouten/hidden-objects", split="train")
sample_path = ds_check[0]["bg_path"]
sample_full_path = os.path.join(original_places_root, sample_path)

print("Sample bg_path:", sample_path)
print("Resolved path:", sample_full_path)
print("Exists:", os.path.exists(sample_full_path))

if not os.path.exists(sample_full_path):
    raise FileNotFoundError(
        f"Sample background image was not found.\n"
        f"Expected: {sample_full_path}\n"
        f"Check that original_places_root is correct."
    )

# --- COLLECT ALL REQUIRED BACKGROUND PATHS ---
needed_paths = set()

for split in splits:
    print(f"\nLoading split: {split}")
    ds = load_dataset("marco-schouten/hidden-objects", split=split)
    print(f"Number of examples in {split}: {len(ds)}")

    # Read only the bg_path column and show progress
    for rel_path in tqdm(ds["bg_path"], total=len(ds), desc=f"Collecting bg_path from {split}"):
        needed_paths.add(rel_path)

print(f"\nTotal unique images needed: {len(needed_paths)}")

# --- COPY OR SYMLINK FILES ---
missing = []
copied = 0
skipped = 0

for rel_path in tqdm(needed_paths, desc="Creating subset folder"):
    src = os.path.join(original_places_root, rel_path)
    dst = os.path.join(subset_places_root, rel_path)

    if not os.path.exists(src):
        missing.append(rel_path)
        continue

    os.makedirs(os.path.dirname(dst), exist_ok=True)

    if os.path.exists(dst):
        skipped += 1
        continue

    if use_symlinks:
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)

    copied += 1

print("\nDone.")
print(f"Subset saved to: {subset_places_root}")
print(f"Copied: {copied}")
print(f"Already existed / skipped: {skipped}")
print(f"Missing files: {len(missing)}")

if missing:
    missing_file = os.path.join(subset_places_root, "missing_files.txt")
    with open(missing_file, "w", encoding="utf-8") as f:
        for path in missing:
            f.write(path + "\n")
    print(f"Missing file list saved to: {missing_file}")