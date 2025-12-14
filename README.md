# Speech Emotion Recognition (CS460G – University of Kentucky)

Classifies speaker emotion from short WAV recordings using:
- a **mel-spectrogram + CNN** pipeline (PyTorch)
- two classical baselines (**KNN** and **SVM**) trained on hand-crafted audio features

This repository is my public fork of our team’s final course project for CS460G.

---

## Results (3-class subset: angry / happy / neutral)

| Model | Accuracy | Macro F1 | Weighted F1 |
|------|----------|----------|-------------|
| KNN | 82.4% | 82.4% | 82.4% |
| SVM | 87.7% | 87.8% | 87.2% |
| CNN (mel-spectrogram) | **93.2%** | **93.2%** | **93.2%** |

Per-class F1 (CNN): **Anger 93.9% | Happy 91.7% | Neutral 94.1%**

If available in this repo:
- `metrics/cnn_outputs/cnn_classification_report.txt`
- `metrics/cnn_outputs/cnn_confusion_matrix.png`

---

## Dataset

We used the Kaggle **Voice Emotion Classification** dataset (not included in this repo).

For CNN experiments, we focused on 3 classes:
- happy (9315 samples)
- anger (9315 samples)
- neutral (7915 samples)

Expected local path (after download/unzip):
- `data/Voice Emotion Dataset/`

Generated spectrograms:
- `data/spectrograms/<label>/<file>.png`

---

## Approach

### CNN pipeline (WAV → mel-spectrogram → CNN)
Preprocessing standardizes audio into a consistent input:
- mono resample to **16,000 Hz**
- enforce **3-second** duration via padding/truncation
- compute **128-band mel-spectrogram**
- export spectrograms to PNG for model training

CNN model (VGG-style, compact):
- input: 1-channel **128×128** spectrogram image
- 3 convolution + pooling blocks (channels 1→16→32→64)
- classifier head with **dropout (p=0.3)**
- trained with Adam (lr 1e-3), batch size 32, 20 epochs
- 70/15/15 train/val/test split with a fixed seed for reproducibility

### Baselines (audio features → classical ML)
KNN and SVM operate on extracted features such as MFCC, delta MFCC, RMS energy, zero-crossing rate, and spectral statistics (centroid/bandwidth/rolloff/contrast).

---

## Repository layout

- `src/` — preprocessing + training scripts (CNN, KNN, SVM)
- `notebooks/` — exploration / experimentation
- `metrics/` — saved evaluation outputs (reports, confusion matrices)
- `models/` — saved models/checkpoints
- `data/` — expected dataset location + generated artifacts (raw Kaggle data not tracked)

---

## Quickstart (high level)

> Script names may vary—see `src/` for the exact filenames.

1) Create an environment and install dependencies (typical packages):
- numpy, pandas, matplotlib
- scikit-learn
- librosa
- torch, torchvision
- pillow

2) Download the Kaggle dataset and unzip into:
- `data/Voice Emotion Dataset/`

3) Run the WAV → spectrogram preprocessing script
4) Train/evaluate the CNN script (writes metrics to `metrics/`)
5) Run KNN and SVM baseline scripts

---

## My contributions (Kevin Dawson-Fischer)

- Designed and implemented the **mel-spectrogram preprocessing pipeline** from raw WAV files (fixed sample rate/duration, export to PNG).
- Implemented the **PyTorch CNN pipeline**, including:
  - custom dataset + loaders
  - CNN architecture
  - training loop w/ validation monitoring + best-checkpoint restore
  - test-set evaluation + metric generation (classification report + confusion matrix)
- Helped define the reduced **3-class** problem (anger/happy/neutral) while keeping the code extendable to additional classes.

---

## Report

Full write-up:
- `docs/Speech_Emotion_Recognition_Report.pdf`
