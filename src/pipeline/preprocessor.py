import os
from typing import Tuple, Union
import cv2
import numpy as np
import tensorflow as tf

from config.settings import config

class ImagePreprocessor:
    """
    Robust Preprocessing Pipeline for Building Structural Imagery.
    Ensures RGB color-space conformance and model-specific normalization.
    """

    @staticmethod
    def load_and_preprocess(
        image_input: Union[str, bytes, np.ndarray],
        target_size: Tuple[int, int] = config.IMAGE_SIZE,
        mode: str = "mobilenet"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads and prepares an image for inference or model training.

        Args:
            image_input: File path (str), raw image bytes, or OpenCV numpy array.
            target_size: Desired spatial dimension (width, height), defaults to (224, 224).
            mode: Normalization strategy:
                - 'mobilenet': tf.keras.applications.mobilenet_v2.preprocess_input ([-1, 1]).
                - 'div255': Basic float32 scaling to [0, 1].

        Returns:
            Tuple of:
                - batch_input (np.ndarray): Tensor of shape (1, H, W, 3) ready for model feed.
                - rgb_display (np.ndarray): Original-scale uint8 RGB array (H, W, 3) for visualizers.
        """
        img_bgr = None

        if isinstance(image_input, (str, os.PathLike)):
            path_str = str(image_input)
            if not os.path.isfile(path_str):
                raise FileNotFoundError(f"Image file does not exist: {path_str}")
            img_bgr = cv2.imread(path_str)
            if img_bgr is None:
                raise ValueError(f"OpenCV could not decode image at path: {path_str}")

        elif isinstance(image_input, (bytes, bytearray)):
            if len(image_input) == 0:
                raise ValueError("Image bytes buffer is empty.")
            nparr = np.frombuffer(image_input, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError("Failed to decode image from byte payload.")

        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input.copy()

        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Decoded image matrix is empty or invalid.")

        # Resize to target geometry
        img_resized = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)

        # Standardize color space to 3-channel RGB
        if len(img_resized.shape) == 2:
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        elif img_resized.shape[2] == 4:
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGRA2RGB)
        else:
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # Apply target normalization
        if mode == "mobilenet":
            normalized = tf.keras.applications.mobilenet_v2.preprocess_input(img_rgb.astype("float32"))
        else:
            normalized = img_resized.astype("float32") / 255.0

        batch_input = np.expand_dims(normalized, axis=0)
        return batch_input, img_rgb
