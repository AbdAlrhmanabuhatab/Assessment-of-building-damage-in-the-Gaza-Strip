import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ─────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────
DATASET_DIR = "dataset"
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 3
BATCH_SIZE = 32
EPOCHS = 10
CLASS_LABELS = {"safe": 0, "repair": 1, "demolish": 2}

# ─────────────────────────────────────────────
# 2. Data Preparation (Same as before)
# ─────────────────────────────────────────────
print("[INFO] Loading images for Fine-Tuning...")
images, labels = [], []
for class_name, label in CLASS_LABELS.items():
    folder = os.path.join(DATASET_DIR, class_name)
    if not os.path.exists(folder): continue
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(cv2.resize(img, IMAGE_SIZE))
            labels.append(label)

X = np.array(images, dtype="float32") / 255.0
y = tf.keras.utils.to_categorical(np.array(labels), num_classes=NUM_CLASSES)
X, y = shuffle(X, y, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ─────────────────────────────────────────────
# 3. Load Model & Unfreeze Top Layers
# ─────────────────────────────────────────────
print("[INFO] Loading the previously trained model...")
model = tf.keras.models.load_model("building_damage_model.h5")

print("[INFO] Unfreezing the last 20 layers of MobileNetV2...")
base_model = model.layers[0] # MobileNetV2 is the first layer
base_model.trainable = True

# Freeze all layers EXCEPT the last 20
for layer in base_model.layers[:-20]:
    layer.trainable = False

# ─────────────────────────────────────────────
# 4. Recompile with a VERY LOW learning rate
# ─────────────────────────────────────────────
# Crucial step: 1e-5 means 0.00001 (very small steps so we don't ruin existing knowledge)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ─────────────────────────────────────────────
# 5. Data Augmentation & Training
# ─────────────────────────────────────────────
datagen = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.1,
    height_shift_range=0.1, zoom_range=0.15,
    horizontal_flip=True, fill_mode="nearest"
)
datagen.fit(X_train)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
    ModelCheckpoint("building_damage_model_finetuned.h5", monitor="val_accuracy", save_best_only=True, verbose=1)
]

print("\n[INFO] Starting Fine-Tuning Training...")
model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    callbacks=callbacks
)

print("\n[SUCCESS] Fine-Tuning complete! Saved as 'building_damage_model_finetuned.h5'")