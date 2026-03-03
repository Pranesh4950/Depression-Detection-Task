import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import cv2
import csv
from tensorflow.keras.models import load_model

# ===============================
# PATHS (MATCH YOUR SETUP)
# ===============================
MODEL_PATH = r"E:\nlp_task\depression_cnn_model.h5"

TEST_AUDIO_PATH = r"E:\nlp_task\dataset\Test-set-tamil\Test-set-tamil"

TEMP_IMG_PATH = r"E:\nlp_task\temp_img"
CSV_OUTPUT_PATH = r"E:\nlp_task\submission.csv"

IMG_SIZE = 128

# ===============================
# CREATE TEMP FOLDER
# ===============================
os.makedirs(TEMP_IMG_PATH, exist_ok=True)

print("TEST_AUDIO_PATH exists:", os.path.exists(TEST_AUDIO_PATH))
print("MODEL exists:", os.path.exists(MODEL_PATH))

# ===============================
# LOAD MODEL
# ===============================
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# ===============================
# AUDIO → MEL SPECTROGRAM
# ===============================
def audio_to_mel(audio_path, save_path):
    y, sr = librosa.load(audio_path, duration=5)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(3,3))
    librosa.display.specshow(mel_db, sr=sr)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()

# ===============================
# PREDICTION LOOP
# ===============================
results = []

for file in os.listdir(TEST_AUDIO_PATH):
    if file.lower().endswith(".wav"):
        audio_path = os.path.join(TEST_AUDIO_PATH, file)

        img_name = os.path.splitext(file)[0] + ".png"
        img_path = os.path.join(TEMP_IMG_PATH, img_name)

        # Convert audio to image
        audio_to_mel(audio_path, img_path)

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print("❌ Failed to read image for:", file)
            continue

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image / 255.0
        image = image.reshape(1, IMG_SIZE, IMG_SIZE, 3)

        # Predict
        prediction = model.predict(image, verbose=0)[0][0]

        if prediction >= 0.5:
            label = 1
            status = "Depressed"
        else:
            label = 0
            status = "Not Depressed"

        results.append([file, label, status])
        print(f"{file} → {status}")

# ===============================
# SAVE CSV (CODABENCH READY)
# ===============================
with open(CSV_OUTPUT_PATH, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["file_name", "label", "status"])
    writer.writerows(results)

print("✅ submission.csv generated at:", CSV_OUTPUT_PATH)
