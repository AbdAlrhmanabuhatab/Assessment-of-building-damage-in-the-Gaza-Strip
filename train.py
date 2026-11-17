import cv2
import numpy as np
import os
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D                   # ← NEW
from tensorflow.keras.applications import MobileNetV2                        # ← NEW
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATASET_DIR = "dataset"
IMAGE_SIZE  = (224, 224)
IMAGE_SHAPE = (224, 224, 3)
NUM_CLASSES = 3
EPOCHS      = 10
BATCH_SIZE  = 32
MODEL_PATH  = "building_damage_model.h5"

CLASS_LABELS = {
    "safe":     0,
    "repair":   1,
    "demolish": 2,
}

tf.random.set_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# STEP 1 – LOAD, RESIZE & LABEL THE IMAGES
# ─────────────────────────────────────────────
print("[INFO] Loading images...")
images = []
labels = []

for class_name, label in CLASS_LABELS.items():
    class_folder = os.path.join(DATASET_DIR, class_name)

    if not os.path.exists(class_folder):
        print(f"[WARNING] Folder not found, skipping: {class_folder}")
        continue

    print(f"  → Loading images from '{class_folder}' (Label: {label})")

    for filename in os.listdir(class_folder):
        file_path = os.path.join(class_folder, filename)
        image = cv2.imread(file_path)

        if image is None:
            continue

        image = cv2.resize(image, IMAGE_SIZE)
        images.append(image)
        labels.append(label)

# ─────────────────────────────────────────────
# STEP 2 – CONVERT TO NUMPY ARRAYS & NORMALIZE
# ─────────────────────────────────────────────
X = np.array(images, dtype="float32") / 255.0
y = np.array(labels, dtype="int32")

X, y = shuffle(X, y, random_state=42)

print("\n=============================================")
print(f"[INFO] X shape : {X.shape}")
print(f"[INFO] y shape : {y.shape}")
print("=============================================\n")

# ─────────────────────────────────────────────
# STEP 3 – ONE-HOT ENCODING & SPLIT
# ─────────────────────────────────────────────
y_cat = to_categorical(y, num_classes=NUM_CLASSES)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ─────────────────────────────────────────────
# STEP 4 – TRANSFER LEARNING WITH MobileNetV2  ← REPLACED
# ─────────────────────────────────────────────
#
# Why MobileNetV2?
#   MobileNetV2 was pre-trained on ImageNet (1.4 M images, 1 000 classes).
#   Its convolutional layers already know how to detect edges, textures,
#   shapes, and structural patterns — exactly what we need for buildings.
#   Instead of learning all of that from our small dataset, we simply
#   "borrow" those weights and only train a small classification head.
#
# include_top=False   → drops ImageNet's 1 000-class output layer so we
#                       can attach our own 3-class head instead.
# weights="imagenet"  → loads the pre-trained weights (downloads on first run).
# trainable=False     → freezes all 154 base layers so their weights are NOT
#                       updated during training. This prevents the valuable
#                       ImageNet knowledge from being overwritten by our tiny
#                       dataset, and makes training much faster.

print("[INFO] Loading MobileNetV2 base model with ImageNet weights …")

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=IMAGE_SHAPE,
)

# Freeze every layer in the base — only our new head will be trained.
base_model.trainable = False

print(f"[INFO] Base model layers    : {len(base_model.layers)}")
print(f"[INFO] Trainable parameters : {base_model.count_params():,}  (all frozen)")

# Build the full model:
#
#   MobileNetV2 (frozen)          → rich feature maps  (7×7×1280)
#   GlobalAveragePooling2D        → collapses spatial dims → 1280-D vector
#                                   (lighter than Flatten; reduces overfitting)
#   Dense(128, relu)              → learns building-damage-specific combinations
#   Dropout(0.3)                  → mild regularisation on a small head
#   Dense(3, softmax)             → one probability per class

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
    ModelCheckpoint(filepath=MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
]

# ─────────────────────────────────────────────
# STEP 5 – DATA AUGMENTATION & TRAINING
# ─────────────────────────────────────────────
print("\n[INFO] Starting training with Data Augmentation …\n")

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

datagen.fit(X_train)

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
)

# ─────────────────────────────────────────────
# STEP 6 – EVALUATE & SAVE
# ─────────────────────────────────────────────
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\n[RESULT] Validation loss     : {val_loss:.4f}")
print(f"[RESULT] Validation accuracy : {val_acc * 100:.2f} %")

model.save(MODEL_PATH)
print(f"\n[INFO] Final Model saved → {MODEL_PATH}")