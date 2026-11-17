import os
import sys
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# Add project root to sys.path
BASE_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_PATH))

from config.settings import config
from src.models.mobilenet import DamageClassifier

def load_partition(split_dir: str):
    images, labels = [], []
    for class_name, label_idx in config.CLASS_LABELS.items():
        folder = os.path.join(split_dir, class_name)
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, config.IMAGE_SIZE)
            images.append(img_resized)
            labels.append(label_idx)
    return np.array(images, dtype="float32"), np.array(labels, dtype="int32")

def run_training_pipeline():
    print("=" * 65)
    print("  TWO-STAGE TRAINING & FINE-TUNING PIPELINE (MobileNetV2)")
    print("=" * 65)

    train_dir = os.path.join(str(config.SPLIT_DATASET_DIR), "train")
    val_dir = os.path.join(str(config.SPLIT_DATASET_DIR), "val")
    save_path = os.path.join(str(config.WEIGHTS_DIR), "building_damage_model_v2.keras")

    os.makedirs(str(config.WEIGHTS_DIR), exist_ok=True)

    print("\n[INFO] Loading training split...")
    X_train_raw, y_train_raw = load_partition(train_dir)
    print(f"  • Training samples  : {len(X_train_raw)}")

    print("[INFO] Loading validation split...")
    X_val_raw, y_val_raw = load_partition(val_dir)
    print(f"  • Validation samples: {len(X_val_raw)}")

    # Preprocessing with MobileNetV2 formula ([-1, 1])
    X_train = tf.keras.applications.mobilenet_v2.preprocess_input(X_train_raw)
    X_val = tf.keras.applications.mobilenet_v2.preprocess_input(X_val_raw)

    y_train = tf.keras.utils.to_categorical(y_train_raw, num_classes=config.NUM_CLASSES)
    y_val = tf.keras.utils.to_categorical(y_val_raw, num_classes=config.NUM_CLASSES)

    # Balanced class weights with safety bias on Demolish
    weights = compute_class_weight("balanced", classes=np.unique(y_train_raw), y=y_train_raw)
    class_weight_dict = {i: float(weights[i]) for i in range(len(weights))}
    class_weight_dict[config.CLASS_LABELS["demolish"]] *= 1.15

    print("\n[INFO] Applied Class Weights:")
    for c_name, c_idx in config.CLASS_LABELS.items():
        print(f"  • {c_name:<10}: {class_weight_dict[c_idx]:.3f}")

    # Data augmentation pipeline
    datagen = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        shear_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest"
    )

    # Initialize model
    print("\n[INFO] Building transfer learning architecture...")
    model = DamageClassifier.build_architecture()

    # Stage 1: Feature Extraction (Base frozen)
    print("\n[INFO] Stage 1: Fitting classification head...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        datagen.flow(X_train, y_train, batch_size=16),
        validation_data=(X_val, y_val),
        epochs=12,
        class_weight=class_weight_dict,
        callbacks=[EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1)],
        verbose=1
    )

    # Stage 2: Fine-Tuning top 25 layers
    print("\n[INFO] Stage 2: Fine-tuning top 25 convolutional layers...")
    base_model = model.layers[0]
    base_model.trainable = True
    for layer in base_model.layers[:-25]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        ModelCheckpoint(save_path, monitor="val_accuracy", save_best_only=True, verbose=1)
    ]

    model.fit(
        datagen.flow(X_train, y_train, batch_size=16),
        validation_data=(X_val, y_val),
        epochs=15,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # Save checkpoint
    model.save(save_path)
    # Also sync to root for backward compatibility
    model.save("building_damage_model_v2.keras")
    print(f"\n[SUCCESS] Model successfully trained and saved to '{save_path}'!")

if __name__ == "__main__":
    run_training_pipeline()
