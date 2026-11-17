import os
import sys

# Ensure UTF-8 output encoding for Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

from preprocess import CLASS_NAMES, CLASS_LABELS

TRAIN_DIR = os.path.join("dataset_split", "train")
VAL_DIR = os.path.join("dataset_split", "val")
MODEL_SAVE_PATH = "building_damage_model_v2.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 3

tf.random.set_seed(42)
np.random.seed(42)

def load_split_images(data_dir):
    images, labels = [], []
    for class_name, label_idx in CLASS_LABELS.items():
        folder = os.path.join(data_dir, class_name)
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, IMAGE_SIZE)
            images.append(img_resized)
            labels.append(label_idx)
    return np.array(images, dtype="float32"), np.array(labels, dtype="int32")

def train_advanced_model():
    print("=" * 60)
    print("  ADVANCED TWO-STAGE TRAINING PIPELINE (MobileNetV2)")
    print("=" * 60)

    # 1. Load Train and Validation sets
    print("\n[INFO] Loading training dataset...")
    X_train_raw, y_train_raw = load_split_images(TRAIN_DIR)
    print(f"  • Training samples  : {len(X_train_raw)}")

    print("[INFO] Loading validation dataset...")
    X_val_raw, y_val_raw = load_split_images(VAL_DIR)
    print(f"  • Validation samples: {len(X_val_raw)}")

    # 2. Preprocess using official MobileNetV2 formula ([-1, 1])
    X_train = preprocess_input(X_train_raw)
    X_val = preprocess_input(X_val_raw)

    y_train = tf.keras.utils.to_categorical(y_train_raw, num_classes=NUM_CLASSES)
    y_val = tf.keras.utils.to_categorical(y_val_raw, num_classes=NUM_CLASSES)

    # 3. Compute Class Weights for cost-sensitive fairness
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train_raw),
        y=y_train_raw
    )
    class_weight_dict = {i: float(weights[i]) for i in range(len(weights))}
    # Give slight extra penalty (1.15x) to Demolish class to prioritize structural safety
    class_weight_dict[CLASS_LABELS["demolish"]] *= 1.15
    print("\n[INFO] Balanced Class Weights calculated:")
    for c_name, c_idx in CLASS_LABELS.items():
        print(f"  • {c_name:<10}: {class_weight_dict[c_idx]:.3f}")

    # 4. Advanced Data Augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        shear_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest"
    )

    # 5. Build Model Architecture
    print("\n[INFO] Initializing MobileNetV2 Base...")
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        Dropout(0.4),
        Dense(NUM_CLASSES, activation="softmax")
    ])

    # ── STAGE 1: Train Head Only ──────────────────
    print("\n[INFO] === STAGE 1: Training Classification Head (Base Frozen) ===")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    stage1_callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1)
    ]

    model.fit(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=12,
        class_weight=class_weight_dict,
        callbacks=stage1_callbacks,
        verbose=1
    )

    # ── STAGE 2: Fine-Tuning Top Layers ──────────
    print("\n[INFO] === STAGE 2: Fine-Tuning Top 25 Layers of MobileNetV2 ===")
    base_model.trainable = True
    for layer in base_model.layers[:-25]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    stage2_callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True, verbose=1)
    ]

    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=15,
        class_weight=class_weight_dict,
        callbacks=stage2_callbacks,
        verbose=1
    )

    # Save final model
    model.save(MODEL_SAVE_PATH)
    print(f"\n[SUCCESS] Upgraded model saved to '{MODEL_SAVE_PATH}'!")

if __name__ == "__main__":
    train_advanced_model()
