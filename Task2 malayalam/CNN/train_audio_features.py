import os
import librosa
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ===============================
# PATHS — CHANGE ONLY IF NEEDED
# ===============================
BASE_PATH = r"E:\nlp_task\malayalam\Malayalam"

DEP_TRAIN = os.path.join(BASE_PATH, "Depressed")
NONDEP_TRAIN = os.path.join(BASE_PATH, "Non_depressed")

MODEL_PATH = r"E:\nlp_task\malayalam\malayalam_audio_model.pkl"

# ===============================
# FEATURE EXTRACTION FUNCTION
# ===============================
def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, duration=5)

        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        features = np.hstack([mfcc, chroma, contrast, zcr])
        return features

    except Exception as e:
        print("❌ Error reading:", audio_path, e)
        return None


# ===============================
# LOAD DATA
# ===============================
X = []
y = []

print("DEP PATH:", DEP_TRAIN)
print("NONDEP PATH:", NONDEP_TRAIN)

# ---- DEPRESSED = label 1 ----
for root, dirs, files in os.walk(DEP_TRAIN):
    for file in files:
        if file.lower().endswith((".wav", ".mp3")):
            path = os.path.join(root, file)
            feat = extract_features(path)
            if feat is not None:
                X.append(feat)
                y.append(1)

# ---- NON DEPRESSED = label 0 ----
for root, dirs, files in os.walk(NONDEP_TRAIN):
    for file in files:
        if file.lower().endswith((".wav", ".mp3")):
            path = os.path.join(root, file)
            feat = extract_features(path)
            if feat is not None:
                X.append(feat)
                y.append(0)

# ===============================
# CHECK DATA
# ===============================
X = np.array(X)
y = np.array(y)

print("\n✅ Total training samples:", len(X))

if len(X) == 0:
    print("❌ No audio files found. Check folder paths.")
    exit()

print("Feature shape:", X.shape)

# ===============================
# TRAIN MODEL (SVM)
# ===============================
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", probability=True))
])

print("\n⏳ Training model...")
model.fit(X, y)

# ===============================
# SAVE MODEL
# ===============================
joblib.dump(model, MODEL_PATH)

print("\n✅ Model trained successfully")
print("💾 Model saved at:", MODEL_PATH)
