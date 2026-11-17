import os
import shutil
import argparse
from typing import Dict, List
from sklearn.model_selection import train_test_split

from config.settings import config

def partition_dataset(
    src_dir: str = str(config.RAW_DATASET_DIR),
    dest_dir: str = str(config.SPLIT_DATASET_DIR),
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    print("=" * 60)
    print("  STRATIFIED DATASET PARTITIONING PIPELINE")
    print("=" * 60)

    all_files: List[str] = []
    all_labels: List[str] = []

    for class_name in config.CLASS_LABELS.keys():
        folder = os.path.join(src_dir, class_name)
        if not os.path.exists(folder):
            print(f"[WARNING] Missing class folder: {folder}")
            continue

        files = [
            os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
        ]
        for f in files:
            all_files.append(f)
            all_labels.append(class_name)
        print(f"  • {class_name:<10}: {len(files)} samples")

    total = len(all_files)
    print(f"\n[INFO] Total scanned images: {total}")

    train_files, temp_files, train_labels, temp_labels = train_test_split(
        all_files, all_labels,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=all_labels
    )

    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels,
        test_size=0.5,
        random_state=seed,
        stratify=temp_labels
    )

    splits = {
        "train": (train_files, train_labels),
        "val": (val_files, val_labels),
        "test": (test_files, test_labels)
    }

    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)

    stats: Dict[str, Dict[str, int]] = {s: {c: 0 for c in config.CLASS_LABELS.keys()} for s in splits}

    for split_name, (files, labels) in splits.items():
        for f_path, label in zip(files, labels):
            target_folder = os.path.join(dest_dir, split_name, label)
            os.makedirs(target_folder, exist_ok=True)
            shutil.copy2(f_path, target_folder)
            stats[split_name][label] += 1

    print("\n" + "=" * 60)
    print(f"  SUMMARY: DISJOINT STRATIFIED PARTITIONS")
    print("=" * 60)
    print(f"{'Split':<10} | {'Safe':<8} | {'Repair':<8} | {'Demolish':<8} | {'Total':<8}")
    print("-" * 60)
    for s in ("train", "val", "test"):
        t_count = sum(stats[s].values())
        print(f"{s:<10} | {stats[s]['safe']:<8} | {stats[s]['repair']:<8} | {stats[s]['demolish']:<8} | {t_count:<8}")
    print("=" * 60)
    print(f"[SUCCESS] Clean partitions established in '{dest_dir}'\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Partition dataset into Train/Val/Test splits.")
    parser.add_argument("--seed", type=int, default=42, help="Random split seed.")
    args = parser.parse_args()
    partition_dataset(seed=args.seed)
