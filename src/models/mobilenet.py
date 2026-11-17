import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential, Model

from config.settings import config, BASE_DIR
from src.pipeline.preprocessor import ImagePreprocessor

class DamageClassifier:
    """
    Production MobileNetV2 Classifier for Structural Damage Assessment.
    Manages architecture initialization, checkpoint loading, and inference.
    """

    def __init__(self, model_path: Optional[str] = None, norm_mode: Optional[str] = None):
        self.model = None
        self.model_path = None
        self.norm_mode = norm_mode
        self._load_model(model_path)

    def _load_model(self, explicit_path: Optional[str] = None):
        if explicit_path and os.path.exists(explicit_path):
            self.model = tf.keras.models.load_model(explicit_path)
            self.model_path = explicit_path
            if not self.norm_mode:
                self.norm_mode = "mobilenet" if explicit_path.endswith(".keras") else "div255"
            return

        # Auto-detect best available weights from candidates
        for candidate_rel, mode in config.MODEL_CANDIDATES:
            candidate_abs = BASE_DIR / candidate_rel
            if candidate_abs.exists():
                self.model = tf.keras.models.load_model(str(candidate_abs))
                self.model_path = str(candidate_abs)
                self.norm_mode = mode
                return

        raise FileNotFoundError(
            f"No valid model weights found in {[c[0] for c in config.MODEL_CANDIDATES]}"
        )

    @classmethod
    def build_architecture(
        cls,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = config.NUM_CLASSES
    ) -> Sequential:
        """
        Builds transfer learning MobileNetV2 architecture with regularization.
        """
        base_model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape
        )
        base_model.trainable = False

        model = Sequential([
            base_model,
            GlobalAveragePooling2D(name="gap"),
            BatchNormalization(name="batch_norm"),
            Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4), name="dense_features"),
            Dropout(0.4, name="dropout"),
            Dense(num_classes, activation="softmax", name="predictions")
        ], name="gaza_damage_mobilenetv2")

        return model

    def predict(self, image_input: Any) -> Dict[str, Any]:
        """
        Runs complete inference pipeline on an image input.
        """
        img_tensor, rgb_vis = ImagePreprocessor.load_and_preprocess(
            image_input,
            target_size=config.IMAGE_SIZE,
            mode=self.norm_mode
        )

        preds = self.model.predict(img_tensor, verbose=0)[0]
        class_idx = int(np.argmax(preds))
        confidence = float(preds[class_idx]) * 100.0
        predicted_class = config.CLASS_NAMES[class_idx]

        probabilities = {
            config.CLASS_NAMES[i]: round(float(preds[i]) * 100.0, 2)
            for i in range(len(config.CLASS_NAMES))
        }

        return {
            "prediction": predicted_class,
            "class_index": class_idx,
            "confidence": round(confidence, 2),
            "probabilities": probabilities,
            "model_path": os.path.basename(self.model_path),
            "norm_mode": self.norm_mode,
            "tensor_input": img_tensor,
            "rgb_visualization": rgb_vis,
            "advisory": config.ENGINEERING_ADVISORIES.get(predicted_class, {})
        }
