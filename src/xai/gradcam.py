import base64
from typing import Optional, Tuple
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

class GradCAMVisualizer:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) Visualizer.
    Produces visual explanations for decisions made by convolutional neural networks.
    """

    @staticmethod
    def generate_heatmap(
        img_array: np.ndarray,
        model: tf.keras.Model,
        last_conv_layer_name: Optional[str] = None,
        pred_index: Optional[int] = None
    ) -> np.ndarray:
        """
        Computes 2D normalized Grad-CAM activation heatmap in range [0, 1].

        Args:
            img_array: Preprocessed input tensor (1, H, W, 3).
            model: Trained Keras model.
            last_conv_layer_name: Optional name of the target conv layer.
            pred_index: Target category index (defaults to max predicted class).

        Returns:
            np.ndarray: 2D float32 heatmap with shape (H, W) in [0.0, 1.0].
        """
        sub_model = None
        if len(model.layers) > 0 and hasattr(model.layers[0], "layers"):
            sub_model = model.layers[0]

        if last_conv_layer_name is None:
            search_target = sub_model if sub_model is not None else model
            for layer in reversed(search_target.layers):
                if "conv" in layer.name.lower() or "out_relu" in layer.name.lower():
                    last_conv_layer_name = layer.name
                    break

        if last_conv_layer_name is None:
            last_conv_layer_name = "out_relu"

        if sub_model is not None:
            conv_layer = sub_model.get_layer(last_conv_layer_name)
            model_inp = sub_model.input if hasattr(sub_model, "input") else sub_model.inputs
            feature_extractor = Model(inputs=model_inp, outputs=[conv_layer.output, sub_model.output])

            with tf.GradientTape() as tape:
                conv_outputs, base_features = feature_extractor(img_array)
                tape.watch(conv_outputs)

                x = base_features
                for layer in model.layers[1:]:
                    x = layer(x)
                preds = x

                if pred_index is None:
                    pred_index = tf.argmax(preds[0])
                class_channel = preds[:, pred_index]

            grads = tape.gradient(class_channel, conv_outputs)
        else:
            conv_layer = model.get_layer(last_conv_layer_name)
            model_inp = model.input if hasattr(model, "input") else model.inputs
            grad_model = Model(inputs=model_inp, outputs=[conv_layer.output, model.output])

            with tf.GradientTape() as tape:
                conv_outputs, preds = grad_model(img_array)
                tape.watch(conv_outputs)
                if pred_index is None:
                    pred_index = tf.argmax(preds[0])
                class_channel = preds[:, pred_index]

            grads = tape.gradient(class_channel, conv_outputs)

        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the spatial channels by importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Rectified Linear Activation (focus only on positive impact)
        heatmap = tf.maximum(heatmap, 0.0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val

        return heatmap.numpy()

    @staticmethod
    def overlay_heatmap(
        rgb_img: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.55,
        colormap: int = cv2.COLORMAP_JET
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Blends a Grad-CAM heatmap with an RGB base image.

        Returns:
            Tuple of:
                - blended_rgb (np.ndarray): Overlaid image (H, W, 3) uint8.
                - colored_heatmap_rgb (np.ndarray): Pure JET thermal map (H, W, 3) uint8.
        """
        heatmap_resized = cv2.resize(heatmap, (rgb_img.shape[1], rgb_img.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        colored_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

        blended = np.float32(colored_rgb) * alpha + np.float32(rgb_img) * (1.0 - alpha)
        blended = np.clip(blended, 0, 255).astype("uint8")

        return blended, colored_rgb

    @staticmethod
    def to_base64_data_uri(rgb_array: np.ndarray, quality: int = 90) -> str:
        """Encodes an RGB uint8 array to a base64 Data URI."""
        bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not success:
            return ""
        b64_str = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"
