import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import cv2
import csv
from tensorflow.keras.models import load_model

# ===============================
# PATHS — MALAYALAM SETUP
# ===============================
BASE_PATH = r"E:\nlp_task\malayalam"

MODEL_PATH = os.path.join(BASE_PATH, "depression_cnn_model.h5")
TEST_AUDIO_PATH = os.path.join(BASE_PATH, "Test_set_mal")
TEMP_IMG_PATH = os.path.join(BASE_PATH, "temp_img")
CSV_OUTPUT_PATH = os.path.join(BASE_PATH, "submission.csv")

IMG_SIZE = 128
VALID_EXT = (".wav", ".mp3", ".flac", ".ogg")

# ===============================
# SETUP
# ===============================
os.makedirs(TEMP_IMG_PATH, exist_ok=True)

print("TEST_AUDIO_PATH exists:", os.path.exists(TEST_AUDIO_PATH))
print("MODEL exists:", os.path.exists(MODEL_PATH))

if not os.path.exists(MODEL_PATH):
    print("❌ Model not found — run train_cnn.py first")
    exit()

# ===============================
# LOAD MODEL
# ===============================
model = load_model(MODEL_PATH)
print("✅ Model loaded")

# ===============================
# AUDIO → MEL SPECTROGRAM
# ===============================
def audio_to_mel(audio_path, save_path):
    try:
        y, sr = librosa.load(audio_path, duration=5)

        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        plt.figure(figsize=(3, 3))
        librosa.display.specshow(mel_db, sr=sr)
        plt.axis("off")

        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()
        return True

    except Exception as e:
        print("❌ Spectrogram failed:", audio_path)
        return False


# ===============================
# COLLECT + NUMERIC SORT FILES
# ===============================
all_files = []

for root, dirs, files in os.walk(TEST_AUDIO_PATH):
    for file in files:
        if file.lower().endswith(VALID_EXT):
            all_files.append(os.path.join(root, file))

def get_num(path):
    name = os.path.basename(path)
    num = ''.join(filter(str.isdigit, name))
    return int(num) if num else 0

all_files.sort(key=get_num)

print("Total valid audio files found:", len(all_files))

# ===============================
# PREDICTION LOOP
# ===============================
results = []
processed = 0
skipped = 0

for audio_path in all_files:

    file = os.path.basename(audio_path)
    img_name = os.path.splitext(file)[0] + ".png"
    img_path = os.path.join(TEMP_IMG_PATH, img_name)

    ok = audio_to_mel(audio_path, img_path)
    if not ok:
        skipped += 1
        continue

    image = cv2.imread(img_path)
    if image is None:
        print("❌ Image read failed:", file)
        skipped += 1
        continue

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    image = image.reshape(1, IMG_SIZE, IMG_SIZE, 3)

    pred = model.predict(image, verbose=0)[0]
    label = int(np.argmax(pred))

    status = "Depressed" if label == 1 else "Not Depressed"

    print(f"{file} → {status}")

    results.append([file, label, status])
    processed += 1


# ===============================
# SAVE CSV
# ===============================
with open(CSV_OUTPUT_PATH, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["file_name", "label", "status"])
    writer.writerows(results)

# ===============================
# SUMMARY
# ===============================
print("\n======================")
print("Processed:", processed)
print("Skipped:", skipped)
print("======================")

print("\n✅ submission.csv saved at:", CSV_OUTPUT_PATH)
