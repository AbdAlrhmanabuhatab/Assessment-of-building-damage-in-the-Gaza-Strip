import os
import sys
import base64
import requests
import numpy as np

# Ensure UTF-8 output encoding for Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from preprocess import load_and_preprocess_image, CLASS_NAMES
from gradcam import get_gradcam_heatmap, overlay_gradcam

BASE_URL = "http://127.0.0.1:8000"

def run_full_audit():
    print("=" * 70)
    print("       COMPREHENSIVE SYSTEM INTEGRITY & END-TO-END AUDIT")
    print("=" * 70)

    passed_tests = 0
    total_tests = 0

    def check(name, condition, details=""):
        nonlocal passed_tests, total_tests
        total_tests += 1
        if condition:
            passed_tests += 1
            print(f"  [PASS] {name} {details}")
            return True
        else:
            print(f"  [FAIL] {name} - {details}")
            return False

    # -------------------------------------------------------------
    # 1. CORE FILES & ARTIFACTS AUDIT
    # -------------------------------------------------------------
    print("\n--- [Phase 1: Files, Models & Datasets Integrity] ---")
    
    check("Model V2 Exists", os.path.exists("building_damage_model_v2.keras"), "building_damage_model_v2.keras")
    check("Requirements File Exists", os.path.exists("requirements.txt"), "requirements.txt")
    check("Split Dataset Exists", os.path.exists("dataset_split/test/demolish"), "dataset_split partitioned")
    check("Web Frontend Files", all(os.path.exists(f) for f in ["static/index.html", "static/style.css", "static/app.js"]), "All 3 web assets present")
    check("Sample Images Exist", all(os.path.exists(f"static/samples/{c}.jpg") for c in ["demolish", "repair", "safe"]), "3 preset benchmark images present")

    # -------------------------------------------------------------
    # 2. MODEL INFERENCE & GRAD-CAM MATHEMATICAL INTEGRITY
    # -------------------------------------------------------------
    print("\n--- [Phase 2: Model & Grad-CAM Mathematical Audit] ---")
    try:
        model = tf.keras.models.load_model("building_damage_model_v2.keras")
        check("Model Load", model is not None, f"Input shape: {model.input_shape}, Output shape: {model.output_shape}")

        dummy_input = np.random.uniform(-1, 1, (1, 224, 224, 3)).astype("float32")
        preds = model.predict(dummy_input, verbose=0)[0]
        check("Probabilities Sum to 1.0", abs(np.sum(preds) - 1.0) < 1e-4, f"Sum = {np.sum(preds):.6f}")
        check("Output Has 3 Classes", len(preds) == 3, f"Classes: {CLASS_NAMES}")

        heatmap = get_gradcam_heatmap(dummy_input, model, pred_index=int(np.argmax(preds)))
        check("Grad-CAM Output Range", 0.0 <= np.min(heatmap) and np.max(heatmap) <= 1.0 + 1e-6, f"Min: {np.min(heatmap):.3f}, Max: {np.max(heatmap):.3f}")
        check("Grad-CAM No NaNs", not np.isnan(heatmap).any(), "Heatmap matrix is strictly numeric")
    except Exception as e:
        check("Model & Grad-CAM Evaluation", False, f"Exception: {e}")

    # -------------------------------------------------------------
    # 3. FASTAPI BACKEND & NETWORK INTEGRATION
    # -------------------------------------------------------------
    print("\n--- [Phase 3: Network & API End-to-End Audit] ---")
    
    # 3.1 Health check
    try:
        r_health = requests.get(f"{BASE_URL}/api/health", timeout=5)
        check("API Health Endpoint", r_health.status_code == 200, f"Status: {r_health.status_code}")
        h_data = r_health.json()
        check("API Model Loaded Correctly", h_data.get("model") == "building_damage_model_v2.keras", f"Model: {h_data.get('model')}")
    except Exception as e:
        check("API Health Endpoint Reachable", False, str(e))

    # 3.2 Samples API
    try:
        r_samples = requests.get(f"{BASE_URL}/api/samples", timeout=5)
        check("API Samples Endpoint", r_samples.status_code == 200 and len(r_samples.json()) == 3, f"Returned {len(r_samples.json())} presets")
    except Exception as e:
        check("API Samples Endpoint", False, str(e))

    # 3.3 Predict API with real images
    sample_tests = ["demolish.jpg", "repair.jpg", "safe.jpg"]
    for sample_name in sample_tests:
        sample_path = os.path.join("static", "samples", sample_name)
        try:
            with open(sample_path, "rb") as f:
                r_pred = requests.post(
                    f"{BASE_URL}/api/predict",
                    files={"file": (sample_name, f, "image/jpeg")},
                    timeout=15
                )
            
            p_data = r_pred.json()
            is_ok = r_pred.status_code == 200
            check(f"Predict Endpoint ({sample_name})", is_ok, f"Result: {p_data.get('prediction')} ({p_data.get('confidence')}%)")
            
            has_overlay = p_data.get("gradcam_overlay_base64", "").startswith("data:image/jpeg;base64,")
            has_orig = p_data.get("original_image_base64", "").startswith("data:image/jpeg;base64,")
            check(f"Grad-CAM Base64 Generated ({sample_name})", has_overlay and has_orig, "Valid Data URIs returned")

            protocols = p_data.get("advisory", {}).get("protocols", [])
            check(f"Protocols Populated ({sample_name})", len(protocols) >= 3, f"{len(protocols)} safety steps")

        except Exception as e:
            check(f"Predict Endpoint ({sample_name})", False, str(e))

    # 3.4 Error Handling & Edge Cases
    print("\n--- [Phase 4: Security & Edge Case Handling] ---")
    try:
        r_text = requests.post(
            f"{BASE_URL}/api/predict",
            files={"file": ("test.txt", b"This is not an image", "text/plain")},
            timeout=5
        )
        check("Reject Non-Image Files (HTTP 400)", r_text.status_code == 400, f"Code: {r_text.status_code}")

        r_empty = requests.post(
            f"{BASE_URL}/api/predict",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
            timeout=5
        )
        check("Reject Empty Files (HTTP 400)", r_empty.status_code == 400, f"Code: {r_empty.status_code}")
    except Exception as e:
        check("Edge Case Handling", False, str(e))

    # -------------------------------------------------------------
    # 4. WEB FRONTEND DOM & SCRIPT BINDING AUDIT
    # -------------------------------------------------------------
    print("\n--- [Phase 5: Web UI Frontend Integrity] ---")
    try:
        r_index = requests.get(f"{BASE_URL}/", timeout=5)
        html = r_index.text
        check("Serve Root HTML", r_index.status_code == 200, f"Size: {len(html)} bytes")
        
        required_dom_ids = [
            "drop-zone", "file-input", "analyze-btn", "clear-image-btn",
            "samples-container", "verdict-card", "verdict-title", "confidence-value",
            "gradcam-overlay-img", "protocols-list", "model-name-badge"
        ]
        all_dom_found = all(f'id="{i}"' in html for i in required_dom_ids)
        check("All Critical DOM IDs Bound in HTML", all_dom_found, f"{len(required_dom_ids)}/11 elements verified")

        r_js = requests.get(f"{BASE_URL}/static/app.js", timeout=5)
        check("Static app.js HTTP 200", r_js.status_code == 200, f"{len(r_js.text)} bytes")

        r_css = requests.get(f"{BASE_URL}/static/style.css", timeout=5)
        check("Static style.css HTTP 200", r_css.status_code == 200, f"{len(r_css.text)} bytes")

    except Exception as e:
        check("Frontend Integrity", False, str(e))

    # -------------------------------------------------------------
    # SUMMARY REPORT
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  SYSTEM AUDIT FINAL RESULT: {passed_tests} / {total_tests} TESTS PASSED")
    if passed_tests == total_tests:
        print("  >>> SYSTEM HEALTH: 100% OPERATIONAL & PRODUCTION READY! <<<")
    else:
        print(f"  >>> WARNING: {total_tests - passed_tests} TEST(S) FAILED! <<<")
    print("=" * 70 + "\n")

    return passed_tests == total_tests

if __name__ == "__main__":
    success = run_full_audit()
    sys.exit(0 if success else 1)
