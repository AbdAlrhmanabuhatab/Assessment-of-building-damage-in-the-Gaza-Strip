import os
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
BASE_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_PATH))

from config.settings import config
from src.models.mobilenet import DamageClassifier
from src.xai.gradcam import GradCAMVisualizer

def run_cli_prediction(image_path: str, model_path: str = None, explain: bool = False):
    if not os.path.exists(image_path):
        print(f"\n[ERROR] File not found: '{image_path}'\n")
        return

    classifier = DamageClassifier(model_path=model_path)
    result = classifier.predict(image_path)

    pred = result["prediction"]
    conf = result["confidence"]
    probs = result["probabilities"]
    adv = result["advisory"]

    print("\n" + "=" * 60)
    print("  GAZA STRUCTURAL DAMAGE ASSESSMENT REPORT (CLI)")
    print("=" * 60)
    print(f"  Target Image   : {os.path.basename(image_path)}")
    print(f"  Model Version  : {result['model_path']} (Norm: {result['norm_mode']})")
    print(f"  Final Decision : {pred.upper()}")
    print(f"  Confidence     : {conf:.2f}%")
    print("-" * 60)
    print("  Class Probability Distribution:")
    for name, p_val in probs.items():
        bar = "#" * int(p_val // 4)
        print(f"    • {name:<10}: {p_val:6.2f}% | {bar}")
    print("-" * 60)
    print(f"  Engineering Status: {adv.get('status_badge', 'N/A')}")
    print(f"  Action Summary    : {adv.get('action_summary', 'N/A')}")
    print("=" * 60 + "\n")

    if explain:
        print("[INFO] Computing Grad-CAM Explainable AI Visualization...")
        heatmap = GradCAMVisualizer.generate_heatmap(
            img_array=result["tensor_input"],
            model=classifier.model,
            pred_index=result["class_index"]
        )
        blended, _ = GradCAMVisualizer.overlay_heatmap(result["rgb_visualization"], heatmap)
        out_cam = f"gradcam_cli_{os.path.basename(image_path)}"
        import cv2
        cv2.imwrite(out_cam, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        print(f"[SUCCESS] Grad-CAM overlaid visualization saved to '{out_cam}'\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify building damage from an image.")
    parser.add_argument("image", nargs="?", default="test_image.jpg", help="Path to input image.")
    parser.add_argument("--model", type=str, default=None, help="Optional model checkpoint path.")
    parser.add_argument("--explain", action="store_true", help="Generate Grad-CAM explanation image.")
    args = parser.parse_args()

    run_cli_prediction(args.image, model_path=args.model, explain=args.explain)
