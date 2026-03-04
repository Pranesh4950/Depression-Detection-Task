# SERENE@DravidianLangTech 2026  
## Comparative Acoustic, Spectrogram, and Transformer-Based Models for Depression Detection in Dravidian Speech

This repository contains the implementation of our system submitted to the **Shared Task on Depression Detection from Malayalam and Tamil Speech Data** at **DravidianLangTech 2026**.

---

## 👥 Team SERENE

TT Pranesh  
KK Thamizhmathi  
S Vigneshwaran  
B Bharathi  

Department of Computer Science and Engineering  
Sri Sivasubramania Nadar College of Engineering  

---

## 📌 Task Description

The task focuses on **binary depression detection** from speech recordings in:

- Tamil  
- Malayalam  

Each utterance is labeled as:
- `Depressed`
- `Non-depressed`

---

# 🧠 System Overview

We implemented **three modeling paradigms**:

---

## 1️⃣ Acoustic Feature-Based Model (SVM)

- Extracted 33-dimensional handcrafted features:
  - MFCC (mean & std)
  - Pitch statistics
  - Energy statistics
  - Silence ratio
- Classifier:
  - SVM (RBF kernel)
  - C = 10
  - Gamma = scale
  - Class weight = balanced


## 2️⃣ Spectrogram-Based CNN Model

- Audio converted to Mel-spectrogram images
- Resized to 128×128
- CNN architecture:
  - Conv2D → MaxPooling
  - Conv2D → MaxPooling
  - Dense → Dropout
  - Sigmoid output


## 3️⃣ Transformer-Based Model (Tamil)

- Audio transcribed using **Whisper (medium)**
- Text cleaned and tokenized
- Fine-tuned **XLM-RoBERTa-base**
- Learning rate: 2e-5
- Epochs: 6
- Batch size: 8


# 📊 Results Summary

### Tamil

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| Acoustic + SVM | 1.00 | 1.00 |
| Spectrogram + CNN | 1.00 | 1.00 |
| Whisper + XLM-R | 0.96 | 0.96 |

### Malayalam

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| Acoustic + SVM | 1.00 | 1.00 |
| Spectrogram + CNN | 1.00 | 1.00 |


# 🛠 Installation

### Create environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
