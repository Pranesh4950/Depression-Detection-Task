import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# MALAYALAM DATASET PATH
# ===============================
BASE_PATH = r"E:\nlp_task\malayalam\Malayalam"

# 🔹 AUDIO SOURCE FOLDERS
DEP_AUDIO = os.path.join(BASE_PATH, "Depressed")
NONDEP_AUDIO = os.path.join(BASE_PATH, "Non_depressed")

# 🔹 OUTPUT IMAGE FOLDERS
OUT_BASE = r"E:\nlp_task\malayalam"
DEP_IMG = os.path.join(OUT_BASE, "depressed_img")
NONDEP_IMG = os.path.join(OUT_BASE, "non_depressed_img")

# create folders if missing
os.makedirs(DEP_IMG, exist_ok=True)
os.makedirs(NONDEP_IMG, exist_ok=True)

print("DEP_AUDIO exists:", os.path.exists(DEP_AUDIO))
print("NONDEP_AUDIO exists:", os.path.exists(NONDEP_AUDIO))
print("DEP_IMG:", DEP_IMG)
print("NONDEP_IMG:", NONDEP_IMG)


# ===============================
# MEL SPECTROGRAM FUNCTION
# ===============================
def audio_to_mel(audio_path, save_path):
    try:
        y, sr = librosa.load(audio_path, duration=5)

        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        plt.figure(figsize=(3, 3))
        librosa.display.specshow(mel_db, sr=sr)
        plt.axis("off")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    except Exception as e:
        print("❌ Error:", audio_path)
        print(e)


# ===============================
# CONVERT DEPRESSED AUDIO
# ===============================
print("\n🎧 Converting Depressed audio...")
for root, dirs, files in os.walk(DEP_AUDIO):
    for file in files:
        if file.lower().endswith((".wav", ".mp3")):
            audio_path = os.path.join(root, file)
            img_name = os.path.splitext(file)[0] + ".png"
            img_path = os.path.join(DEP_IMG, img_name)

            audio_to_mel(audio_path, img_path)


# ===============================
# CONVERT NON DEPRESSED AUDIO
# ===============================
print("\n🎧 Converting Non-depressed audio...")
for root, dirs, files in os.walk(NONDEP_AUDIO):
    for file in files:
        if file.lower().endswith((".wav", ".mp3")):
            audio_path = os.path.join(root, file)
            img_name = os.path.splitext(file)[0] + ".png"
            img_path = os.path.join(NONDEP_IMG, img_name)

            audio_to_mel(audio_path, img_path)


print("\n✅ Spectrogram images generated successfully.")
