"""
===============================================================================
Title       : convert_to_spectrograms.py
Project     : Speech Emotion Recognition
Authors     : Alex Dimayuga, Kevin Dawson-Fischer 
Created     : November 2, 2025
Description : 
    This script converts WAV files to Mel-spectrograms to train/test/validate
    our machine learning models like CNN, KNN, and SVM.

Dependencies:
    - os
    - glob
    - librosa
    - soundfile
    - numpy
    - tqdm 

Usage:
    Run this on WAV files to convert them to 2D spectrogram

    Example:
        python convert_to_spectrograms.py 

Notes:
    - TODO:
===============================================================================
"""

# Imports
# =============================================================================
import os 
import glob
import librosa
import librosa.display 
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
# =============================================================================

# =============================================================================
#Configuration
TARGET_SAMPLE_RATE = 16000  #audio sample rate
TARGET_SECONDS = 3        #fixed duration (in seconds)
N_MELS = 128            #number of mel bands
FMIN = 20              #minimum frequency in Hz
FMAX = 8000           #maximum frequency in Hz
N_FFT = 2048         #length of the FFT window
HOP_LENGTH = 512     #number of samples between successive frames

#Image export
FIG_DPI = 100 #2.24 in * 100 dpi = 224 pixels
FIG_SIZE = (2.24, 2.24)  #size in inches

#I/O roots
DEFAULT_IN_DIR = "data\Voice Emotion Dataset"
DEFAULT_OUT_DIR = "data/spectrograms"
# =============================================================================

# Global Variables
# =============================================================================
WAV_DATA_PATH = "../data/Voice Emotion Dataset"
EMOTIONS_TO_USE = ["anger", "happy", "neutral"]
# =============================================================================


# Helper Functions
# =============================================================================
# def loadDataset():
#     file_paths = []
#     labels = [] 

#     for emotion in EMOTIONS_TO_USE:
#         emotion_folder = os.path.join(WAV_DATA_PATH, emotion)
#         if os.path.isdir(emotion_folder):
#             for wav_file in glob.glob(os.path.join(emotion_folder, "*.wav")):
#                 file_paths.append(wav_file)
#                 labels.append(emotion)
    
#     print("Number of files loaded: ", len(file_paths))
#     print("Example file: ", file_paths[0], "Label: ", labels[0])
#     print("Duplicate files?: ", len(file_paths) != len(set(file_paths)))

#     return file_paths, labels

def loadDataset(in_root: Path = Path(DEFAULT_IN_DIR)) -> tuple[list[Path], list[str]]:
    """
    Walk the input tree and return (wav_paths, labels)
    Assume label == immediate parent folder name (e.g., raw/happy/*.wav)
    """

    wav_paths: list[Path] = []
    labels: list[str] = []

    #glob recursively: raw/<label>/**/*.wav
    pattern = str(in_root / "**" / "*.wav")
    for p in glob.glob(pattern, recursive = True):
        path = Path(p)
        label = path.parent.name #one level up is emotion folder
        wav_paths.append(path)
        labels.append(label)
    return wav_paths, labels

# def wavToMelSpectrogram(wav_path):
#     n_mels = 128    # Number of Mel frequency bins
#     fmax = 8000     # 8Khz since typical speech freq is around 90-255 Hz (for adult male/females)

#     # Load audio
#     y, sr = librosa.load(wav_path, sr=None)  # sr=None preserves original sample rate

#     # Convert to Mel-spectrogram
#     S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)

#     # Convert to log scale (decibels)
#     S_dB = librosa.power_to_db(S, ref=np.max)

#     return S_dB

def wavToMelSpectrogram(wav_path: Path,
                        sr: int = TARGET_SAMPLE_RATE,
                        seconds: float = TARGET_SECONDS,
                        n_mels: int = N_MELS,
                        fmin: int = FMIN, fmax: int = FMAX,
                        n_fft: int = N_FFT, hop_length: int = HOP_LENGTH) -> np.ndarray:
    
    """
    Framework-preserving function: load -> fixed length -> mel dB
    """

    y, sr = load_fixed_audio(wav_path, sr = sr, seconds = seconds)
    s_db = mel_db(y, sr, n_mels = n_mels, fmin = fmin, fmax = fmax,
                  n_fft = n_fft, hop_length = hop_length)
    return s_db


def load_fixed_audio(wav_path: Path,
                     sr: int = TARGET_SAMPLE_RATE,
                     seconds: float = TARGET_SECONDS) -> tuple[np.ndarray, int]:
    """
    Load mono audio file at specified sample rate and fixed duration.
    """

    n_samples = int(sr * seconds)
    y, _ = librosa.load(str(wav_path), sr = sr, mono=True)
    if len(y) < n_samples:
        y = np.pad(y, (0, n_samples - len(y)))
    else:
        y = y[:n_samples]
    
    return y, sr
    
def mel_db(y: np.ndarray,
           sr: int,
           n_mels: int = N_MELS,
           fmin: int = FMIN,
           fmax: int = FMAX,
           n_fft: int = N_FFT,
           hop_length: int = HOP_LENGTH) -> np.ndarray:
    """
    Compute mel-spectrogram in decibels (dB)
    """

    S = librosa.feature.melspectrogram(
        y = y, sr = sr, n_mels = n_mels, fmin = fmin, fmax = fmax,
        n_fft = n_fft, hop_length = hop_length, power = 2.0)
    S_db = librosa.power_to_db(S, ref = np.max)
    return S_db

def save_spec_png(S_db: np.ndarray, out_path: Path,
                  sr: int = TARGET_SAMPLE_RATE,
                  fmin: int = FMIN, fmax: int = FMAX,
                  figsize = FIG_SIZE, dpi: int = FIG_DPI) -> None:
    """
    Save mel-spectrogram (dB) to a clean PNG (no axes), fixed pixel size
    """

    Path(out_path).parent.mkdir(parents = True, exist_ok = True)
    plt.figure(figsize = figsize, dpi = dpi)
    librosa.display.specshow(S_db, sr = sr, fmin = fmin, fmax = fmax, cmap = "magma")
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.savefig(out_path, bbox_inches = "tight", pad_inches = 0)
    plt.close()



# =============================================================================

# Main Function
# =============================================================================
def main():
    # # 1.) Load dataset
    # loadDataset()
    parser = argparse.ArgumentParser(
        description = "Convert WAVs to mel-spectrogram PNGs with fixed shapes."
    )
    parser.add_argument("--in", dest = "in_dir", type = str, default = DEFAULT_IN_DIR,
                        help = "Input root directory containing <label>/*.wav (default: data/raw)")
    parser.add_argument("--out", dest = "out_dir", type = str, default = DEFAULT_OUT_DIR,
                        help = "Output root for spectrogram PNGs (default: data/spectrograms)")
    parser.add_argument("--seconds", type = float, default = TARGET_SECONDS,
                        help = f"Fixed audio length in seconds (default: {TARGET_SECONDS})")
    parser.add_argument("--sr", type = int, default = TARGET_SAMPLE_RATE,
                        help = f"Resample sample rate (default: {TARGET_SAMPLE_RATE})")
    parser.add_argument("--n-mels", type = int, default = N_MELS,
                        help = f"Number of mel bands (default: {N_MELS})")
    parser.add_argument("--fmin", type = int, default = FMIN,
                        help = f"Lowest frequency (Hz) (default: {FMIN})")
    parser.add_argument("--fmax", type = int, default = FMAX,
                        help = f"Highest frequency (Hz) (default: {FMAX})")
    parser.add_argument("--n-fft", type = int, default = N_FFT,
                        help = f"FFT window size (default: {N_FFT})")
    parser.add_argument("--hop", type = int, default = HOP_LENGTH,
                        help = f"Hop length (default: {HOP_LENGTH})")
    
    args = parser.parse_args()
    in_root = Path(args.in_dir)
    out_root = Path(args.out_dir)

    wav_paths, labels = loadDataset(in_root)

    if not wav_paths:
        print("No WAV files found in input directory.")
        return

    print(f"Found {len(wav_paths)} WAV files. Saving PNG's under: {out_root.resolve()}")

    for wav_path, label in tqdm(list(zip(wav_paths, labels)), desc = "Converting"):
        try:
            S_db = wavToMelSpectrogram(
                wav_path,
                sr = args.sr,
                seconds = args.seconds,
                n_mels = args.n_mels,
                fmin = args.fmin, fmax = args.fmax,
                n_fft = args.n_fft, hop_length = args.hop,
            )
            out_path = out_root / label / (wav_path.stem + ".png")
            save_spec_png(S_db, out_path, sr = args.sr, fmin = args.fmin, fmax = args.fmax)
        except Exception as e:
            print(f"[ERROR] {wav_path}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
# =============================================================================
