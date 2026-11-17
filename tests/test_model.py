import os
import pytest
import numpy as np
from src.models.mobilenet import DamageClassifier
from config.settings import config

def test_model_loading_and_inference():
    classifier = DamageClassifier()
    assert classifier.model is not None
    assert classifier.norm_mode in ("mobilenet", "div255")

    # Generate synthetic input
    dummy_input = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    result = classifier.predict(dummy_input)

    assert result["prediction"] in config.CLASS_NAMES
    assert 0.0 <= result["confidence"] <= 100.0
    assert len(result["probabilities"]) == 3
    prob_sum = sum(result["probabilities"].values())
    assert 99.0 <= prob_sum <= 101.0

def test_architecture_builder():
    model = DamageClassifier.build_architecture()
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 3)
