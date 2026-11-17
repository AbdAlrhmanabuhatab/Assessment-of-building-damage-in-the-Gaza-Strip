import os
import sys

# Ensure UTF-8 output encoding for Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from preprocess import load_and_preprocess_image, CLASS_NAMES, CLASS_LABELS

TEST_DIR = os.path.join("dataset_split", "test")

def evaluate_model(model_path, mode="mobilenet", cm_filename="confusion_matrix_v2.png"):
    if not os.path.exists(model_path):
        print(f"[ERROR] Model '{model_path}' not found!")
        return

    print("=" * 65)
    print(f"  EVALUATION FOR: {os.path.basename(model_path)} (Mode: {mode})")
    print("=" * 65)
    model = tf.keras.models.load_model(model_path)

    images = []
    y_true = []

    for class_name, label_idx in CLASS_LABELS.items():
        class_folder = os.path.join(TEST_DIR, class_name)
        if not os.path.exists(class_folder):
            continue

        for filename in os.listdir(class_folder):
            filepath = os.path.join(class_folder, filename)
            try:
                img_input, _ = load_and_preprocess_image(filepath, mode=mode)
                images.append(img_input[0])
                y_true.append(label_idx)
            except Exception as e:
                print(f"[WARNING] Skipping {filepath}: {e}")

    X_test = np.array(images, dtype="float32")
    y_true = np.array(y_true, dtype="int32")

    raw_preds = model.predict(X_test, verbose=0)
    y_pred = np.argmax(raw_preds, axis=1)

    # Classification report
    print("\n" + "-" * 65)
    print(f"  CLASSIFICATION REPORT on Independent Test Set (N={len(y_true)})")
    print("-" * 65)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES
    )
    plt.title(f"Confusion Matrix ({os.path.basename(model_path)} - Test Set)", fontsize=11, pad=12)
    plt.ylabel("Ground Truth (Actual)", fontsize=11)
    plt.xlabel("Model Prediction", fontsize=11)
    plt.tight_layout()
    plt.savefig(cm_filename, dpi=150)
    plt.close()
    print(f"[SUCCESS] Confusion Matrix saved to '{cm_filename}'")

    # Safety Risk Metric
    catastrophic_underestimation = np.sum((y_true == 2) & (y_pred == 0))
    print("-" * 65)
    print(f"  [SAFETY METRIC] Demolished classified as Safe: {catastrophic_underestimation}")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    v2_model = "building_damage_model_v2.keras"
    if os.path.exists(v2_model):
        evaluate_model(v2_model, mode="mobilenet", cm_filename="confusion_matrix_v2.png")
    else:
        evaluate_model("building_damage_model_finetuned.h5", mode="div255", cm_filename="confusion_matrix_test.png")
