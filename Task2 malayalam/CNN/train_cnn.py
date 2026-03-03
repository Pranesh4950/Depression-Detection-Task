import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ===============================
# PATHS — MALAYALAM
# ===============================
BASE_PATH = r"E:\nlp_task\malayalam"

DEP_IMG = os.path.join(BASE_PATH, "depressed_img")
NONDEP_IMG = os.path.join(BASE_PATH, "non_depressed_img")

MODEL_PATH = os.path.join(BASE_PATH, "depression_cnn_model.h5")

IMG_SIZE = 128

print("DEP_IMG exists:", os.path.exists(DEP_IMG))
print("NONDEP_IMG exists:", os.path.exists(NONDEP_IMG))

# ===============================
# LOAD IMAGES
# ===============================
X = []
y = []

def load_folder(folder, label):
    for file in os.listdir(folder):
        if file.lower().endswith(".png"):
            path = os.path.join(folder, file)
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(label)

load_folder(DEP_IMG, 1)
load_folder(NONDEP_IMG, 0)

X = np.array(X) / 255.0
y = to_categorical(y, 2)

print("Total images:", len(X))

if len(X) == 0:
    print("❌ No images found — run audio_to_spectrogram.py first")
    exit()

# ===============================
# BUILD CNN
# ===============================
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(2, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ===============================
# TRAIN
# ===============================
print("\n⏳ Training CNN...")
model.fit(X, y, epochs=10, batch_size=16, validation_split=0.2)

# ===============================
# SAVE MODEL
# ===============================
model.save(MODEL_PATH)

print("\n✅ CNN model saved at:", MODEL_PATH)
