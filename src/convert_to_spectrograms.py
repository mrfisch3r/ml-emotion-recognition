"""
===============================================================================
Title       : convert_to_spectrograms.py
Project     : Speech Emotion Recognition
Author      : Alex Dimayuga 
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
# =============================================================================

# Global Variables
# =============================================================================
WAV_DATA_PATH = "../data/Voice Emotion Dataset"
EMOTIONS_TO_USE = ["anger", "happy", "neutral"]
# =============================================================================


# Helper Functions
# =============================================================================
def loadDataset():
    file_paths = []
    labels = [] 

    for emotion in EMOTIONS_TO_USE:
        emotion_folder = os.path.join(WAV_DATA_PATH, emotion)
        if os.path.isdir(emotion_folder):
            for wav_file in glob.glob(os.path.join(emotion_folder, "*.wav")):
                file_paths.append(wav_file)
                labels.append(emotion)
    
    print("Number of files loaded: ", len(file_paths))
    print("Example file: ", file_paths[0], "Label: ", labels[0])
    print("Duplicate files?: ", len(file_paths) != len(set(file_paths)))

    return file_paths, labels

def wavToMelSpectrogram(wav_path):
    n_mels = 128    # Number of Mel frequency bins
    fmax = 8000     # 8Khz since typical speech freq is around 90-255 Hz (for adult male/females)

    # Load audio
    y, sr = librosa.load(wav_path, sr=None)  # sr=None preserves original sample rate

    # Convert to Mel-spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)

    # Convert to log scale (decibels)
    S_dB = librosa.power_to_db(S, ref=np.max)

    return S_dB



# =============================================================================

# Main Function
# =============================================================================
def main():
    # 1.) Load dataset
    loadDataset()

if __name__ == "__main__":
    main()
# =============================================================================
