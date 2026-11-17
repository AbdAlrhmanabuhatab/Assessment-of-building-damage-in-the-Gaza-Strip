import os
import shutil
from sklearn.model_selection import train_test_split

DATASET_SRC = "dataset"
DATASET_DEST = "dataset_split"

CLASSES = ["safe", "repair", "demolish"]
RANDOM_STATE = 42

# Split ratios: 70% Train, 15% Validation, 15% Test
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def split_and_copy():
    all_files = []
    all_labels = []

    print("[INFO] Scanning source dataset...")
    for class_name in CLASSES:
        folder = os.path.join(DATASET_SRC, class_name)
        if not os.path.exists(folder):
            print(f"[WARNING] Folder '{folder}' does not exist!")
            continue

        files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        for f in files:
            all_files.append(f)
            all_labels.append(class_name)
        print(f"  • {class_name:<10}: {len(files)} images")

    total_images = len(all_files)
    print(f"\n[INFO] Total valid images found: {total_images}")

    # First split: Separate Train (70%) and Temp (30% for Val + Test)
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        all_files, all_labels,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_STATE,
        stratify=all_labels
    )

    # Second split: Split Temp 50/50 into Val (15%) and Test (15%)
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=temp_labels
    )

    splits = {
        "train": (train_files, train_labels),
        "val": (val_files, val_labels),
        "test": (test_files, test_labels)
    }

    # Reset destination directory
    if os.path.exists(DATASET_DEST):
        shutil.rmtree(DATASET_DEST)

    print("\n[INFO] Copying images to structured splits...")
    stats = {s: {c: 0 for c in CLASSES} for s in splits}

    for split_name, (files, labels) in splits.items():
        for f_path, label in zip(files, labels):
            dest_dir = os.path.join(DATASET_DEST, split_name, label)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(f_path, dest_dir)
            stats[split_name][label] += 1

    print("\n" + "=" * 55)
    print(f"  DATASET STRATIFIED SPLIT SUMMARY")
    print("=" * 55)
    print(f"{'Split':<10} | {'Safe':<8} | {'Repair':<8} | {'Demolish':<8} | {'Total':<8}")
    print("-" * 55)
    for s in ["train", "val", "test"]:
        row_total = sum(stats[s].values())
        print(f"{s:<10} | {stats[s]['safe']:<8} | {stats[s]['repair']:<8} | {stats[s]['demolish']:<8} | {row_total:<8}")
    print("=" * 55)
    print(f"[SUCCESS] Dataset successfully partitioned into '{DATASET_DEST}'!\n")

if __name__ == "__main__":
    split_and_copy()
