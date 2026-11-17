import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

def get_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    Generates a Grad-CAM heatmap for a given input image and model.
    
    Args:
        img_array: Batch numpy array with shape (1, 224, 224, 3) preprocessed for the model.
        model: Trained Keras model (Sequential or Functional).
        last_conv_layer_name: Optional name of the convolutional layer to extract gradients from.
        pred_index: Optional target class index (defaults to top predicted class).
        
    Returns:
        heatmap: 2D numpy array of size (224, 224) with values in [0, 1].
    """
    # 1. Identify the base model and target conv layer
    # If the model is a Sequential wrapper containing MobileNetV2 as layer 0:
    target_model = model
    sub_model = None

    if len(model.layers) > 0 and hasattr(model.layers[0], "layers"):
        sub_model = model.layers[0]

    if last_conv_layer_name is None:
        # Search for the last Conv2D or DepthwiseConv2D layer in sub_model or model
        search_target = sub_model if sub_model is not None else model
        for layer in reversed(search_target.layers):
            if "conv" in layer.name.lower() or "out_relu" in layer.name.lower():
                last_conv_layer_name = layer.name
                break

    if last_conv_layer_name is None:
        last_conv_layer_name = "out_relu"

    # Build a gradient model that maps the inputs to activations of the conv layer + model output
    if sub_model is not None:
        conv_layer = sub_model.get_layer(last_conv_layer_name)
        # Create intermediate submodel (using .input to prevent Keras 3 structure warnings)
        model_inp = sub_model.input if hasattr(sub_model, "input") else sub_model.inputs
        feature_extractor = Model(inputs=model_inp, outputs=[conv_layer.output, sub_model.output])
        
        with tf.GradientTape() as tape:
            conv_outputs, base_features = feature_extractor(img_array)
            tape.watch(conv_outputs)
            
            # Pass base_features through remaining head layers
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

    # Compute pooled gradients across spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the channels by gradient importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU (we only care about features that positively contribute to the target class)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def overlay_gradcam(rgb_img, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
    """
    Overlays a Grad-CAM heatmap onto an RGB image.
    
    Args:
        rgb_img: Original RGB image (H, W, 3) with values in [0, 255], uint8.
        heatmap: 2D array (H, W) in [0, 1].
        alpha: Heatmap transparency factor (0.0 to 1.0).
        colormap: OpenCV colormap.
        
    Returns:
        superimposed_img: Blended RGB image.
    """
    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (rgb_img.shape[1], rgb_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply colormap (OpenCV returns BGR)
    colored_heatmap_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap_rgb = cv2.cvtColor(colored_heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Blend images
    blended = np.float32(colored_heatmap_rgb) * alpha + np.float32(rgb_img) * (1.0 - alpha)
    blended = np.clip(blended, 0, 255).astype("uint8")
    return blended
