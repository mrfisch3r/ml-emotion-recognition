# Speech Emotion Recognition (CS460G Final Project)

Speech Emotion Recognition using audio signals. We classify **anger / happy / neutral** from voice data using classic ML baselines and a **PyTorch CNN trained on mel-spectrogram images**.

> This repo is a fork of the team repository used for CS460G (University of Kentucky). I’m using this public fork as a portfolio snapshot of the final state + my contributions.

---

## Results (3-class subset)

Overall performance (test set):

| Model | Accuracy |
|------|----------|
| KNN  | 82.4% |
| SVM  | 87.7% |
| **CNN (mel-spectrograms)** | **93.2%** |

Per-class accuracy (CNN):

- Anger: **93.9%**
- Happy: **91.7%**
- Neutral: **94.1%**

CNN classification report (test set):

| Class | Precision | Recall | F1 | Support |
|------|-----------:|-------:|---:|--------:|
| Anger | 0.9451 | 0.9336 | 0.9393 | 1401 |
| Happy | 0.9190 | 0.9143 | 0.9166 | 1389 |
| Neutral | 0.9317 | 0.9505 | 0.9410 | 1191 |
| **Accuracy** |  |  | **0.9319** | 3981 |

---

## What I personally implemented (Kevin Dawson-Fischer)

**Mel-spectrogram preprocessing pipeline**
- Standardizes audio to **16 kHz**, enforces **3-second** duration (pad/truncate)
- Generates **128-band mel-spectrograms** (n_fft=2048, hop_length=512, fmin=20, fmax=8000)
- Exports spectrograms to PNGs under a label folder structure for CNN training

**PyTorch CNN training + evaluation**
- Custom Dataset/DataLoader that reads spectrogram PNGs, converts to grayscale, resizes, normalizes
- CNN model + training loop (Adam, lr=1e-3, 20 epochs, dropout=0.3)
- Saves evaluation artifacts (classification report + confusion matrix) into the repo

---

## Dataset

Source: **Voice Emotion Classification** dataset from Kaggle.  
For experiments, we used a 3-class subset:

- happy (9315)
- anger (9315)
- neutral (7915)

> Note: To keep the repo size reasonable, you should **not** commit the raw Kaggle dataset. Download it locally and run preprocessing to regenerate spectrograms.

---

## Model Overview (CNN)

Input: **1×128×128** spectrogram image

- Conv Block 1: 1→16, 3×3 + ReLU + MaxPool (128→64)
- Conv Block 2: 16→32, 3×3 + ReLU + MaxPool (64→32)
- Conv Block 3: 32→64, 3×3 + ReLU + MaxPool (32→16)
- Flatten → Linear(64×16×16 → 128) + ReLU + Dropout(0.3) → Linear(128 → 3)

Train/val/test split: **70/15/15**, fixed seed for reproducibility.

---

## Repository layout

- `src/` — preprocessing + training/eval scripts
- `notebooks/` — experimentation
- `metrics/` — saved reports/figures
- `models/` — saved model artifacts (if
