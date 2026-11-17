import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ─────────────────────────────────────────────
# Basic Configuration
# ─────────────────────────────────────────────
DATASET_DIR = "dataset"
IMAGE_SIZE = (224, 224)
CLASS_LABELS = {"safe": 0, "repair": 1, "demolish": 2}
CLASS_NAMES = ["Safe", "Repair", "Demolish"]

print("[INFO] Loading the FINETUNED trained model...")
# Loading the newly optimized model
model = tf.keras.models.load_model("building_damage_model_finetuned.h5")

# ─────────────────────────────────────────────
# Read all images for testing
# ─────────────────────────────────────────────
print("[INFO] Reading images to test the finetuned model's performance...")
images = []
true_labels = []

for class_name, label in CLASS_LABELS.items():
    class_folder = os.path.join(DATASET_DIR, class_name)
    if not os.path.exists(class_folder): continue

    for filename in os.listdir(class_folder):
        img_path = os.path.join(class_folder, filename)
        img = cv2.imread(img_path)
        if img is None: continue

        img_resized = cv2.resize(img, IMAGE_SIZE)
        images.append(img_resized)
        true_labels.append(label)

# ─────────────────────────────────────────────
# Preprocessing and Prediction
# ─────────────────────────────────────────────
X = np.array(images, dtype="float32") / 255.0
y_true = np.array(true_labels)

print("[INFO] Generating predictions... please wait.")
predictions = model.predict(X, verbose=0)
y_pred = np.argmax(predictions, axis=1)

# ─────────────────────────────────────────────
# Print report and plot confusion matrix
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("  FINETUNED MODEL CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix - Finetuned Model Performance")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")

print("[INFO] Saving the new confusion matrix image...")
# Saved with a different name to compare both images
plt.savefig("confusion_matrix_finetuned.png")
print("[SUCCESS] Done! Check the 'confusion_matrix_finetuned.png' file in your folder.\n")