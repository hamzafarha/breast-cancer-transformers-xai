"""
Script to generate data/processed/metadata.csv from the raw BUSI dataset.

This replicates the metadata-building logic from 01_dataset_exploration.ipynb
and saves the result so that 02_data_preprocessing.ipynb can load it.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

DATASET_CANDIDATES = [
    DATA_ROOT / "raw" / "Dataset_BUSI_with_GT",
    DATA_ROOT / "raw" / "BUSI" / "Dataset_BUSI_with_GT",
    DATA_ROOT / "raw" / "BUSI",
]
DATASET_ROOT = next((c for c in DATASET_CANDIDATES if c.exists()), DATASET_CANDIDATES[0])

CLASS_NAMES = ["normal", "benign", "malignant"]
CLASS_TO_LABEL = {"normal": 0, "benign": 1, "malignant": 2}


def is_mask_path(path: Path) -> bool:
    return bool(re.search(r"_mask(?:_\d+)?$", path.stem.lower()))


def build_metadata(dataset_root: Path) -> pd.DataFrame:
    mask_lookup: dict[str, list[str]] = defaultdict(list)

    for class_name in CLASS_NAMES:
        class_dir = dataset_root / class_name
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() != ".png":
                continue
            if is_mask_path(path):
                base_stem = re.sub(r"_mask(?:_\d+)?$", "", path.stem, flags=re.IGNORECASE)
                mask_lookup[base_stem].append(str(path.resolve()))

    records = []
    for class_name in CLASS_NAMES:
        class_dir = dataset_root / class_name
        if not class_dir.exists():
            continue
        for image_path in sorted(class_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() != ".png":
                continue
            if is_mask_path(image_path):
                continue
            image_id = image_path.stem
            masks = mask_lookup.get(image_id, [])
            records.append(
                {
                    "image_path": str(image_path.resolve()),
                    "class_name": class_name,
                    "label": CLASS_TO_LABEL[class_name],
                    "image_id": image_id,
                    "mask_paths": json.dumps(masks),
                    "num_masks": len(masks),
                }
            )

    return pd.DataFrame(records)


if __name__ == "__main__":
    print(f"Dataset root: {DATASET_ROOT}")
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_ROOT}")

    df = build_metadata(DATASET_ROOT)
    print(f"Total images: {len(df)}")
    print(df["class_name"].value_counts().to_string())

    out_path = DATA_ROOT / "processed" / "metadata.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")
