import numpy as np
import tensorflow as tf
from src.xai.gradcam import GradCAMVisualizer
from src.models.mobilenet import DamageClassifier

def test_gradcam_computation():
    classifier = DamageClassifier()
    dummy_input = np.random.uniform(-1, 1, (1, 224, 224, 3)).astype("float32")
    heatmap = GradCAMVisualizer.generate_heatmap(dummy_input, classifier.model, pred_index=0)

    assert heatmap.shape == (7, 7) or heatmap.shape == (224, 224) or len(heatmap.shape) == 2
    assert not np.isnan(heatmap).any()
    assert 0.0 <= np.min(heatmap)
    assert np.max(heatmap) <= 1.0 + 1e-5

def test_overlay_and_base64():
    rgb_dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    heatmap_dummy = np.random.uniform(0, 1, (224, 224)).astype("float32")
    blended, thermal = GradCAMVisualizer.overlay_heatmap(rgb_dummy, heatmap_dummy)

    assert blended.shape == (224, 224, 3)
    assert blended.dtype == np.uint8
    assert thermal.shape == (224, 224, 3)

    b64 = GradCAMVisualizer.to_base64_data_uri(blended)
    assert b64.startswith("data:image/jpeg;base64,")
