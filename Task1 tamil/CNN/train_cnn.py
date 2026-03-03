import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ===============================
# PATHS (MATCHING YOUR SETUP)
# ===============================
DATASET_PATH = r"E:\nlp_task\dataset"

DEP_IMG = os.path.join(DATASET_PATH, "depressed_img")
NONDEP_IMG = os.path.join(DATASET_PATH, "non_depressed_img")

IMG_SIZE = 128

X = []
y = []

# ===============================
# LOAD DEPRESSED IMAGES (label = 1)
# ===============================
for img in os.listdir(DEP_IMG):
    img_path = os.path.join(DEP_IMG, img)

    image = cv2.imread(img_path)
    if image is None:
        continue

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    X.append(image)
    y.append(1)

# ===============================
# LOAD NON-DEPRESSED IMAGES (label = 0)
# ===============================
for img in os.listdir(NONDEP_IMG):
    img_path = os.path.join(NONDEP_IMG, img)

    image = cv2.imread(img_path)
    if image is None:
        continue

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    X.append(image)
    y.append(0)

# ===============================
# PREPROCESS
# ===============================
X = np.array(X, dtype="float32") / 255.0
y = np.array(y)

print("Total samples:", len(X))
print("Depressed:", np.sum(y == 1))
print("Non-depressed:", np.sum(y == 0))

# ===============================
# TRAIN / TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# CNN MODEL (CPU FRIENDLY)
# ===============================
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===============================
# TRAIN MODEL
# ===============================
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# ===============================
# SAVE MODEL
# ===============================
model.save(r"E:\nlp_task\depression_cnn_model.h5")
print("✅ Model trained and saved as depression_cnn_model.h5")
