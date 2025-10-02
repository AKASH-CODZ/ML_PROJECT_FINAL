import os
import cv2
import numpy as np
from collections import Counter

DATASET_DIR = "datasetprocessed"  # your folder
OUTPUT_FILE = "dataset_processed_clean.npz"
IMG_SIZE = 224

images, labels = [], []
class_names = [cls for cls in sorted(os.listdir(DATASET_DIR)) if cls != ".DS_Store"]

for label, class_name in enumerate(class_names):
    folder = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(folder):
        continue
    print(f"[INFO] Processing {class_name}...")
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if not (file.lower().endswith(".jpg") or file.lower().endswith(".jpeg") or file.lower().endswith(".png")):
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0
        images.append(img)
        labels.append(label)  

images = np.array(images)
labels = np.array(labels)
np.savez(OUTPUT_FILE, images=images, labels=labels, classes=class_names)
print(f"[INFO] Saved cleaned dataset → {OUTPUT_FILE}")

data = np.load(OUTPUT_FILE, allow_pickle=True)
images, labels, class_names = data["images"], data["labels"], data["classes"]


print(f"Images shape: {images.shape}") 
print(f"Labels shape: {labels.shape}") 
print(f"Number of classes: {len(class_names)}")


print(f"Image dtype: {images.dtype}, min: {images.min()}, max: {images.max()}")
print(f"Labels dtype: {labels.dtype}, unique labels: {np.unique(labels)}")


counter = Counter(labels)
for idx, cls in enumerate(class_names):
    print(f"Class '{cls}' ({idx}): {counter[idx]} images")
