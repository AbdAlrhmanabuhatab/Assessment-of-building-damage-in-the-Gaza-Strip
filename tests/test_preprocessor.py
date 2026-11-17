import os
import pytest
import numpy as np
from src.pipeline.preprocessor import ImagePreprocessor

TEST_IMG_PATH = "assets/test_image.jpg"
if not os.path.exists(TEST_IMG_PATH):
    TEST_IMG_PATH = "test_image.jpg"

def test_preprocess_from_file():
    if not os.path.exists(TEST_IMG_PATH):
        pytest.skip("Test image not available in workspace.")

    tensor, rgb = ImagePreprocessor.load_and_preprocess(TEST_IMG_PATH, mode="mobilenet")
    assert tensor.shape == (1, 224, 224, 3)
    assert rgb.shape == (224, 224, 3)
    # MobileNetV2 preprocessing scales roughly between -1 and 1
    assert tensor.min() >= -1.1
    assert tensor.max() <= 1.1

def test_preprocess_div255():
    if not os.path.exists(TEST_IMG_PATH):
        pytest.skip("Test image not available.")

    tensor, _ = ImagePreprocessor.load_and_preprocess(TEST_IMG_PATH, mode="div255")
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0

def test_preprocess_from_bytes():
    if not os.path.exists(TEST_IMG_PATH):
        pytest.skip("Test image not available.")

    with open(TEST_IMG_PATH, "rb") as f:
        data = f.read()
    tensor, rgb = ImagePreprocessor.load_and_preprocess(data, mode="mobilenet")
    assert tensor.shape == (1, 224, 224, 3)
    assert rgb.shape == (224, 224, 3)

def test_preprocess_empty_bytes_raises():
    with pytest.raises(ValueError):
        ImagePreprocessor.load_and_preprocess(b"")

def test_preprocess_nonexistent_file_raises():
    with pytest.raises(FileNotFoundError):
        ImagePreprocessor.load_and_preprocess("non_existent_path_xyz.jpg")
