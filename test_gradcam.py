import os
import sys

# Ensure UTF-8 output encoding for Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from preprocess import load_and_preprocess_image, CLASS_NAMES
from gradcam import get_gradcam_heatmap, overlay_gradcam

MODEL_PATH = "building_damage_model_v2.keras"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "building_damage_model_finetuned.h5"

NORM_MODE = "mobilenet" if MODEL_PATH.endswith(".keras") else "div255"

def explain_image(image_path, output_path="gradcam_result.png"):
    if not os.path.exists(image_path):
        print(f"[ERROR] Image '{image_path}' not found.")
        return

    print("=" * 60)
    print("  EXPLAINABLE AI: GRAD-CAM STRUCTURAL VISUALIZATION")
    print("=" * 60)
    print(f"[INFO] Loading model: '{MODEL_PATH}' ...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print(f"[INFO] Preprocessing image: '{image_path}' ...")
    img_input, rgb_orig = load_and_preprocess_image(image_path, mode=NORM_MODE)

    # Inference
    preds = model.predict(img_input, verbose=0)[0]
    top_class_idx = int(np.argmax(preds))
    top_class_name = CLASS_NAMES[top_class_idx]
    confidence = float(preds[top_class_idx]) * 100.0

    print(f"[INFO] Prediction: {top_class_name.upper()} ({confidence:.2f}%)")
    print("[INFO] Computing Grad-CAM heatmap for the top class...")

    heatmap = get_gradcam_heatmap(img_input, model, pred_index=top_class_idx)
    overlay = overlay_gradcam(rgb_orig, heatmap, alpha=0.5)

    # Plot side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb_orig)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title(f"Grad-CAM Heatmap ({top_class_name})", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Focus Area Overlay (Decision: {top_class_name})", fontsize=12)
    axes[2].axis("off")

    plt.suptitle(
        f"Structural Damage Explanation: {os.path.basename(image_path)}\n"
        f"Prediction: {top_class_name} | Confidence: {confidence:.2f}%",
        fontsize=14, y=1.02
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[SUCCESS] Grad-CAM visualization saved to '{output_path}'")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    test_img = "test_image.jpg"
    if len(sys.argv) > 1:
        test_img = sys.argv[1]
    explain_image(test_img)
