import cv2
import numpy as np
import tensorflow as tf

IMAGE_SIZE = (224, 224)

CLASS_NAMES = ["Safe", "Repair", "Demolish"]
CLASS_LABELS = {
    "safe": 0,
    "repair": 1,
    "demolish": 2
}
LABEL_TO_CLASS = {v: k for k, v in CLASS_LABELS.items()}

def load_and_preprocess_image(image_input, target_size=IMAGE_SIZE, mode="div255"):
    """
    Loads and preprocesses an image for inference or training.
    
    Args:
        image_input: File path (str) OR raw bytes OR numpy array (OpenCV BGR/RGB).
        target_size: Tuple (width, height), default (224, 224).
        mode: Normalization mode:
            - 'div255': Normalizes pixels to [0, 1] by dividing by 255.0 (used by existing .h5 models).
            - 'mobilenet': Uses tf.keras.applications.mobilenet_v2.preprocess_input ([-1, 1] range).
    
    Returns:
        np.ndarray with shape (1, height, width, 3) ready for model prediction,
        and the original resized RGB image (height, width, 3) for visualization.
    """
    img_bgr = None
    
    if isinstance(image_input, str):
        img_bgr = cv2.imread(image_input)
        if img_bgr is None:
            raise ValueError(f"Failed to load image from path: {image_input}")
    elif isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Failed to decode image from provided byte stream.")
    elif isinstance(image_input, np.ndarray):
        img_bgr = image_input.copy()
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Resize image
    img_resized = cv2.resize(img_bgr, target_size)

    # Convert BGR to RGB for standard processing & visualization
    # Note: If the image was already 3-channel BGR from cv2:
    if len(img_resized.shape) == 2:
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    elif img_resized.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGRA2RGB)
    else:
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # Normalize
    if mode == "mobilenet":
        normalized = tf.keras.applications.mobilenet_v2.preprocess_input(img_rgb.astype("float32"))
    else:
        # div255 (compatible with existing weights)
        normalized = img_resized.astype("float32") / 255.0

    batch_input = np.expand_dims(normalized, axis=0)
    return batch_input, img_rgb
