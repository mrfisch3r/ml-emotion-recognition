"""
===============================================================================
Title       : feature_extraction_for_baselines.py
Project     : Speech Emotion Recognition
Authors     : 
Created     : November 3, 2025
Description : 
    Extracts features, directly from the audio files, such as MFCC and RMS Energy 
    to train/test/validate our baseline models. 

Dependencies:
    librosa
    numpy
    pandas
    os
    glob
    tqdm

Usage:
    python .\feature_extraction_for_baselines.py

    Example:
        python .\feature_extraction_for_baselines.py

Notes:
    - TODO: Improvements and make CLI-friendly 
===============================================================================
"""
# Imports
# =============================================================================
import librosa
import numpy as np
import pandas as pd
import os 
import glob
from tqdm import tqdm
# =============================================================================

# Global Variables
# =============================================================================
WAV_FILE_PATH = "../data/Voice Emotion Dataset" # Change if file locations change
EMOTIONS_TO_USE = ["anger", "happy", "neutral"] # Change if more emotions needed
# =============================================================================

# Helper Functions
# =============================================================================
def retrieveSpectrograms(wav_file_path):
    file_paths = []
    labels = []

    for emotion in EMOTIONS_TO_USE:
        emotion_folder = os.path.join(wav_file_path, emotion)
        if os.path.isdir(emotion_folder):
            for wav_file in glob.glob(os.path.join(emotion_folder, "*.wav")):
                file_paths.append(wav_file)
                labels.append(emotion)
    
    print("=============================================================================")
    print("Number of files loaded: ", len(file_paths))
    print("Example file: ", file_paths[0], "Label: ", labels[0])
    print("Duplicate files?: ", len(file_paths) != len(set(file_paths)))
    print("=============================================================================")
    print()

    return file_paths, labels

def extractFeatures(wav_path, n_mfcc=13, fmin=20, fmax=8000, sr=16000):
    y, sr = librosa.load(wav_path, sr=sr, mono=True)

    # -------- MFCC --------  
    # Description: Mel-Frequency Cepstral Coefficients that captures the timbral and spectral characteristics of audio signals 
    #              by modeling how humans perceive sound. Informs us how the energy in different frequency bands evolves over time.
    #              Summarizes the overall shape of the sound spectrum.
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, fmin=fmin, fmax=fmax)  
    mfcc_mean = np.mean(mfccs, axis=1)  
    mfcc_std = np.std(mfccs, axis=1)  

    # -------- Delta MFCC --------  
    # Description: First-Order derivatives of MFCCs over time, capturing how the spectral features of an audio signal change frame
    #              to frame. Reveals the dynamics of speech and how it evolves. Measures how the shape changes like tracking pitch,
    #              energy, or articulation shifts
    delta_mfccs = librosa.feature.delta(mfccs)  
    delta_mean = np.mean(delta_mfccs, axis=1)  
    delta_std = np.std(delta_mfccs, axis=1)  

    # -------- Zero-Crossing Rate -------- 
    # Description: Measure of how frequently an audio signal changes sign (crosses the zero amplitude axis) within a given time frame.
    #              Reflects the noisiness, sharpness, or percussiveness of a sound  
    zcr = librosa.feature.zero_crossing_rate(y)  
    zcr_mean = np.mean(zcr)  
    zcr_std = np.std(zcr)  

    # -------- RMS Energy --------  
    # Description: Measure of the average power of an audio signal over time. Reflects how loud or intense a sound is. A key feature 
    #              for analyzing speech dynamics, emotion, and musical texture
    rms = librosa.feature.rms(y=y)  
    rms_mean = np.mean(rms)  
    rms_std = np.std(rms)  

    # -------- Spectral Centroid --------  
    # Description: Indicates the "center of mass" of the spectrum. Tells you where the majority of the energy in a sound is concentrated
    #              along the frequency axis. Often used to describe the brightness or sharpness of a sound.
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)  
    spec_cent_mean = np.mean(spec_cent)  
    spec_cent_std = np.std(spec_cent)  

    # -------- Spectral Bandwidth --------  
    # Description: Measures the spread of frequencies around the spectral centroid in an audio signal. Tells you how wide or narrow the
    #              frequency content is. Essentially how much of the spectrum is active at a given moment.
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)  
    spec_bw_mean = np.mean(spec_bw)  
    spec_bw_std = np.std(spec_bw)  

    # -------- Spectral Roll-off --------  
    # Description: Indicates the frequency below a specified percentage (in this case 85%) of the total spectral energy is contained. It 
    #              is a measure how quickly the energy in a signal "rolls off" toward higher frequencies. Essentially tell you how much 
    #              high-frequency is present
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)  
    rolloff_mean = np.mean(rolloff)  
    rolloff_std = np.std(rolloff)  

    # -------- Spectral Contrast -------- 
    # Description: Measures the difference in energy between peaks and valleys in the frequency spectrum across multiple sub-bands. Captures
    #              how tonal vs. noisy a sound is. 
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)  
    spec_contrast_mean = np.mean(spec_contrast, axis=1)  
    spec_contrast_std = np.std(spec_contrast, axis=1)  

    # -------- Concatenate all features into one vector --------  
    feature_vector = np.concatenate([  
        mfcc_mean, mfcc_std,  
        delta_mean, delta_std,  
        [zcr_mean, zcr_std],  
        [rms_mean, rms_std],  
        [spec_cent_mean, spec_cent_std],  
        [spec_bw_mean, spec_bw_std],  
        [rolloff_mean, rolloff_std],  
        spec_contrast_mean, spec_contrast_std  
    ])  

    return feature_vector  

def displayDf(file_path):
    # Load the CSV
    df = pd.read_csv("../data/features.csv")
    
    # Print shape of the DataFrame
    print("=============================================================================")
    print("Shape:", df.shape)
    print()

    # Print label distribution
    print("\nLabel counts:")
    print(df['label'].value_counts())
    print()
    
    # Preview the first 10 rows
    print("\nFirst 10 rows:")
    print(df.head(10))
    print("=============================================================================")
# =============================================================================

# Main Function
# =============================================================================
def main():
    # 1.) Retrieve spectrograms needed 
    file_paths, labels = retrieveSpectrograms(WAV_FILE_PATH)

    # 2.) Extract features from each file
    features = []
    valid_labels = []
    columns = (
    [f"mfcc_mean_{i}" for i in range(13)] +
    [f"mfcc_std_{i}" for i in range(13)] +
    [f"delta_mean_{i}" for i in range(13)] +
    [f"delta_std_{i}" for i in range(13)] +
    ["zcr_mean", "zcr_std", "rms_mean", "rms_std",
     "spec_cent_mean", "spec_cent_std",
     "spec_bw_mean", "spec_bw_std",
     "rolloff_mean", "rolloff_std"] +
    [f"contrast_mean_{i}" for i in range(7)] +
    [f"contrast_std_{i}" for i in range(7)]
    )
    print("=============================================================================")
    print("Feature Extraction...")
    print("=============================================================================")
    for file, label in tqdm(zip(file_paths, labels), total=len(file_paths)):
        try:
            feature_vector = extractFeatures(file)
            if len(feature_vector) == len(columns):
                features.append(feature_vector)
                valid_labels.append(label)
            else:
                print(f"Feature length mismatch in {file}")
        except Exception as e:
            print(f"Error processing {file}: {e}")
    print()

    # 3.) Create Dataframe
    df = pd.DataFrame(features, columns=columns)

    # 4.) Add labels as a new column
    df['label'] = valid_labels

    # 5.) Save as CSV
    print("=============================================================================")
    print("Saving as csv...")
    print("=============================================================================")
    print()
    df.to_csv("../data/features.csv", index=False)

    # 6.) Display dataframe
    displayDf("../data/features.csv")


if __name__ == "__main__":
    main()
# =============================================================================
