import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Add project root to sys.path
BASE_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_PATH))

from config.settings import config
from src.models.mobilenet import DamageClassifier
from src.pipeline.preprocessor import ImagePreprocessor

def evaluate_test_split(model_path: str = None, save_plots: bool = True):
    print("=" * 65)
    print("  INDEPENDENT TEST SET EVALUATION BENCHMARK")
    print("=" * 65)

    classifier = DamageClassifier(model_path=model_path)
    test_dir = str(config.SPLIT_DATASET_DIR / "test")

    print(f"[INFO] Evaluating model: '{os.path.basename(classifier.model_path)}'")
    print(f"[INFO] Test data source: '{test_dir}'")

    images = []
    y_true = []

    for class_name, label_idx in config.CLASS_LABELS.items():
        folder = os.path.join(test_dir, class_name)
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            try:
                inp_tensor, _ = ImagePreprocessor.load_and_preprocess(
                    fpath,
                    target_size=config.IMAGE_SIZE,
                    mode=classifier.norm_mode
                )
                images.append(inp_tensor[0])
                y_true.append(label_idx)
            except Exception as e:
                print(f"[WARNING] Skipping corrupted image {fpath}: {e}")

    X_test = np.array(images, dtype="float32")
    y_true = np.array(y_true, dtype="int32")

    print(f"[INFO] Loaded {len(X_test)} unseen test samples.")
    raw_preds = classifier.model.predict(X_test, verbose=0)
    y_pred = np.argmax(raw_preds, axis=1)

    print("\n" + "-" * 65)
    print("  CLASSIFICATION REPORT (Strict Holdout Test Split)")
    print("-" * 65)
    report = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, digits=4)
    print(report)

    cm = confusion_matrix(y_true, y_pred)

    if save_plots:
        os.makedirs("assets", exist_ok=True)
        plt.figure(figsize=(7, 5.5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES
        )
        plt.title(f"Confusion Matrix ({os.path.basename(classifier.model_path)})", fontsize=11, pad=12)
        plt.ylabel("Ground Truth (Actual)", fontsize=11)
        plt.xlabel("Model Prediction", fontsize=11)
        plt.tight_layout()

        # Save to assets for README and to root for compatibility
        plt.savefig("assets/confusion_matrix.png", dpi=160)
        plt.savefig("confusion_matrix_v2.png", dpi=160)
        plt.close()
        print("[SUCCESS] Confusion Matrix saved to 'assets/confusion_matrix.png'")

    catastrophic_underestimation = np.sum((y_true == 2) & (y_pred == 0))
    print("-" * 65)
    print(f"  [SAFETY METRIC] Demolished Classified as Safe: {catastrophic_underestimation}")
    if catastrophic_underestimation == 0:
        print("  >>> SAFETY COMPLIANCE: 100% (Zero False-Safe errors on Demolish) <<<")
    print("=" * 65 + "\n")

    return report, cm

if __name__ == "__main__":
    evaluate_test_split()
