import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

DATASET_PATH = r"E:\nlp_task\dataset"

# 🔹 AUDIO SOURCE FOLDERS
DEP_AUDIO = os.path.join(DATASET_PATH, "Depressed", "Train_set")
NONDEP_AUDIO = os.path.join(DATASET_PATH, "Non-depressed", "Train_set")

# 🔹 OUTPUT IMAGE FOLDERS
DEP_IMG = os.path.join(DATASET_PATH, "depressed_img")
NONDEP_IMG = os.path.join(DATASET_PATH, "non_depressed_img")

# create folders if missing
os.makedirs(DEP_IMG, exist_ok=True)
os.makedirs(NONDEP_IMG, exist_ok=True)

print("DEP_AUDIO exists:", os.path.exists(DEP_AUDIO))
print("NONDEP_AUDIO exists:", os.path.exists(NONDEP_AUDIO))
print("DEP_IMG created:", DEP_IMG)
print("NONDEP_IMG created:", NONDEP_IMG)


def audio_to_mel(audio_path, save_path):
    try:
        y, sr = librosa.load(audio_path, duration=5)

        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        plt.figure(figsize=(3,3))
        librosa.display.specshow(mel_db, sr=sr)
        plt.axis("off")

        # ensure directory exists again (extra safety)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    except Exception as e:
        print("❌ Error processing:", audio_path)
        print(e)


# 🔹 Convert Depressed
for file in os.listdir(DEP_AUDIO):
    if file.lower().endswith(".wav"):
        audio_path = os.path.join(DEP_AUDIO, file)
        img_name = os.path.splitext(file)[0] + ".png"
        img_path = os.path.join(DEP_IMG, img_name)

        audio_to_mel(audio_path, img_path)


# 🔹 Convert Non-depressed
for file in os.listdir(NONDEP_AUDIO):
    if file.lower().endswith(".wav"):
        audio_path = os.path.join(NONDEP_AUDIO, file)
        img_name = os.path.splitext(file)[0] + ".png"
        img_path = os.path.join(NONDEP_IMG, img_name)

        audio_to_mel(audio_path, img_path)


print("✅ Spectrogram images generated successfully.")
