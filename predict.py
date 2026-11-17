"""
Root Entry Point for CLI Inference.
Delegates to scripts.predict_cli.
"""
import sys
from scripts.predict_cli import run_cli_prediction

if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "test_image.jpg"
    run_cli_prediction(img, explain=True)
