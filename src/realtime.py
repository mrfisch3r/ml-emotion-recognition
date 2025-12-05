"""
===============================================================================
Title       : realtime.py
Project     : Speech Emotion Recognition
Authors     : Braxton Goble
Created     : November 5, 2025
Description : 
    A real-time speech emotion recognition application using pre-trained models:
    CNN, KNN, and SVM. The application captures audio from the microphone,  processes it, and predicts the emotion in real-time.
    The GUI allows users to select the model and displays the predicted emotion along with a spectrogram.

    

Dependencies:
    - Python 3.9+
    - torch, torchvision
    - numpy
    - pillow (PIL)
    - scikit-learn
    - matplotlib
    -tkinter

Usage:
    python realtime.py


Notes:
    - CNN model is fully predicting anger 100% of the time and never moves. Model has been retrained 
        still showing the same issue. Might need to revisit with different hyperparamters I am not sure.
    - KNN works great through and performs as expected.
    - SVM has the same problem as CNN where it predicts only one class all the time. I have not investigated
      this further yet.
===============================================================================
"""
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import torch
import librosa
import sounddevice as sd
from io import BytesIO
import joblib
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ---- SETTINGS ----
SAMPLE_RATE = 16000
CHUNK_DURATION = 3.0  # 3 seconds to match training
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CNN_MODEL_PATH = PROJECT_ROOT / "models" / "cnn" / "cnn_model_2025-12-05.joblib"
KNN_MODEL_PATH = PROJECT_ROOT / "models" / "knn" / "knn_model_2025-11-28.joblib"
SVM_MODEL_PATH = PROJECT_ROOT / "models" / "svm" / "svm_model_2025-11-28.joblib"
EMOTIONS = ["anger", "happy", "neutral"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- CNN MODEL DEFINITION (from cnn_optimized.py) ----
class EmotionCNN(torch.nn.Module):
    """CNN for spectrogram classification - matches training architecture"""
    
    def __init__(self, num_classes: int):
        super().__init__()
        IMG_SIZE = 128  # From training code
        
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        
        reduced_size = IMG_SIZE // 8
        flattened_dim = 64 * reduced_size * reduced_size
        
        # Classifier with 128 hidden units to match saved checkpoint
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(flattened_dim, 128),  
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(128, num_classes),  
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

# ---- LOAD MODELS ----
print(f"Using device: {DEVICE}")

# Check available audio devices
print("\nAvailable audio devices:")
print("="*60)
try:
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            default = " (DEFAULT INPUT)" if i == sd.default.device[0] else ""
            print(f"  [{i}] {device['name']}{default}")
            print(f"      Sample rate: {device['default_samplerate']} Hz")
            print(f"      Input channels: {device['max_input_channels']}")
except Exception as e:
    print(f"Could not query audio devices: {e}")
print("="*60)

# CNN Model
try:
    cnn_model = joblib.load(CNN_MODEL_PATH)
    cnn_model.to(DEVICE)
    cnn_model.eval()
    print("CNN model loaded successfully")
except Exception as e:
    print(f"Could not load CNN model: {e}")
    cnn_model = None

# KNN Model
try:
    knn_model = joblib.load(KNN_MODEL_PATH)
    print("KNN model loaded successfully")
except Exception as e:
    print(f"Could not load KNN model: {e}")
    knn_model = None

# SVM Model
try:
    svm_model = joblib.load(SVM_MODEL_PATH)
    print("SVM model loaded successfully")
except Exception as e:
    print(f"Could not load SVM model: {e}")
    svm_model = None

# ---- AUDIO PREPROCESSING ----
def extract_features_from_audio(audio, sr=SAMPLE_RATE):
    """
    Extract 76 features from audio exactly as done in training.
    This matches the features.csv preprocessing.
    """
    features = []
    
    # 1. MFCCs (13 coefficients) - mean and std = 26 features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    features.extend(mfcc_mean)  # 13 features
    features.extend(mfcc_std)   # 13 features
    
    # 2. Chroma features (12 features)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    features.extend(chroma_mean)  # 12 features
    
    # 3. Mel Spectrogram (20 features)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=20)
    mel_mean = np.mean(mel, axis=1)
    features.extend(mel_mean)  # 20 features
    
    # 4. Spectral Contrast (7 features)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=6)
    contrast_mean = np.mean(contrast, axis=1)
    features.extend(contrast_mean)  # 7 features
    
    # 5. Tonnetz (6 features)
    try:
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
        tonnetz_mean = np.mean(tonnetz, axis=1)
        features.extend(tonnetz_mean)  # 6 features
    except:
        features.extend([0] * 6)  # Fallback if tonnetz fails
    
    # 6. Additional spectral features (2 features)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
    features.append(spectral_centroid)  # 1 feature
    features.append(spectral_rolloff)   # 1 feature
    
    # Total: 13 + 13 + 12 + 20 + 7 + 6 + 2 = 73 features
    # Need 3 more features to reach 76
    
    # 7. Additional features to reach 76
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    rms = np.mean(librosa.feature.rms(y=audio))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
    features.append(zcr)                # 1 feature
    features.append(rms)                # 1 feature
    features.append(spectral_bandwidth) # 1 feature
    
    features = np.array(features)
    
    # Verify we have exactly 76 features
    if len(features) != 76:
        print(f"Warning: Expected 76 features, got {len(features)}")
    
    return features

def preprocess_for_cnn(audio, sr=SAMPLE_RATE):
    """
    Create mel spectrogram for CNN (128x128) matching training preprocessing.
    """
    # Ensure fixed length (3 seconds)
    target_len = int(sr * 3)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]
    
    # Create mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Ensure 128x128 size (matching IMG_SIZE from training)
    if mel_db.shape[1] < 128:
        mel_db = np.pad(mel_db, ((0, 0), (0, 128 - mel_db.shape[1])), mode='constant')
    else:
        mel_db = mel_db[:, :128]
    
    # Normalize to [0, 1] range first (like ToTensor does)
    mel_normalized = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    
    return mel_normalized

# ---- PREDICTION FUNCTIONS ----
def predict_cnn(audio):
    """Predict emotion using CNN model"""
    if cnn_model is None:
        return "CNN model not loaded", np.zeros((128, 128))
    
    try:
        mel = preprocess_for_cnn(audio)
        
        # Normalize using the exact same method as training: (x - 0.5) / 0.5
        # This maps from [0, 1] range to [-1, 1]
        mel_normalized = (mel - 0.5) / 0.5
        
        # Convert to tensor [1, 1, 128, 128]
        tensor = torch.tensor(mel_normalized).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
        
        with torch.no_grad():
            logits = cnn_model(tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item()
        
        print(f"CNN Prediction: {EMOTIONS[pred_idx]} (confidence: {confidence:.2%})")
        print(f"All probabilities: {[f'{EMOTIONS[i]}: {probs[0,i]:.2%}' for i in range(len(EMOTIONS))]}")
        
        return EMOTIONS[pred_idx], mel
    except Exception as e:
        print(f"CNN prediction error: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", np.zeros((128, 128))

def predict_knn(audio):
    """Predict emotion using KNN model"""
    if knn_model is None:
        return "KNN model not loaded", np.zeros((128, 128))
    
    try:
        features = extract_features_from_audio(audio)
        features = features.reshape(1, -1)
        
        # The KNN model has a pipeline with scaler + knn
        if hasattr(knn_model, 'named_steps'):
            # It's a pipeline - transform features through scaler
            scaler = knn_model.named_steps['scaler']
            knn_classifier = knn_model.named_steps['knn']
            features_scaled = scaler.transform(features)
            
            # Get the classes from the training data
            classes = knn_classifier.classes_
            print(f"  KNN classes in model: {classes}")
            
            # Manually compute distances and get prediction
            distances, indices = knn_classifier.kneighbors(features_scaled)
            
            # Get the labels of nearest neighbors
            neighbor_labels = knn_classifier._y[indices[0]]
            print(f"  Nearest neighbor labels: {neighbor_labels}")
            
            # Count occurrences (for uniform weights) or use distances (for distance weights)
            from collections import Counter
            if knn_classifier.weights == 'uniform':
                most_common = Counter(neighbor_labels).most_common(1)[0][0]
            else:
                # Distance-weighted voting
                label_weights = {}
                for label, dist in zip(neighbor_labels, distances[0]):
                    weight = 1.0 / (dist + 1e-8)
                    label_weights[label] = label_weights.get(label, 0) + weight
                most_common = max(label_weights, key=label_weights.get)
            
            print(f"  Raw prediction: {most_common} (type: {type(most_common)})")
            
            # Handle both string and integer predictions
            if isinstance(most_common, (int, np.integer)):
                # It's an integer index, map to emotion
                emotion = EMOTIONS[most_common]
            elif isinstance(most_common, str):
                # It's already a string
                emotion = most_common.lower()
            else:
                # Try to convert to string
                emotion_str = str(most_common).lower()
                # Check if it's a digit string
                if emotion_str.isdigit():
                    idx = int(emotion_str)
                    if 0 <= idx < len(EMOTIONS):
                        emotion = EMOTIONS[idx]
                    else:
                        emotion = "neutral"
                else:
                    emotion = emotion_str
        else:
            # Direct model (not a pipeline)
            pred = knn_model.predict(features)[0]
            if isinstance(pred, (int, np.integer)):
                emotion = EMOTIONS[pred]
            else:
                emotion = str(pred).lower()
        
        # Ensure the emotion is in our valid emotions list
        if emotion not in [e.lower() for e in EMOTIONS]:
            print(f"Warning: Unknown emotion '{emotion}', defaulting to neutral")
            emotion = "neutral"
        
        print(f"KNN Prediction: {emotion}")
        
        # Generate mel spectrogram for visualization
        mel = preprocess_for_cnn(audio)
        
        return emotion, mel
    except Exception as e:
        print(f"KNN prediction error: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", np.zeros((128, 128))

def predict_svm(audio):
    """Predict emotion using SVM model"""
    if svm_model is None:
        return "SVM model not loaded", np.zeros((128, 128))
    
    try:
        features = extract_features_from_audio(audio)
        features = features.reshape(1, -1)
        
        pred = svm_model.predict(features)[0]
        
        # Model returns string labels directly
        emotion = str(pred).lower()
        
        # Ensure the emotion is in our valid emotions list
        if emotion not in [e.lower() for e in EMOTIONS]:
            print(f"Warning: Unknown emotion '{emotion}', defaulting to neutral")
            emotion = "neutral"
        
        print(f"SVM Prediction: {emotion}")
        
        # Generate mel spectrogram for visualization
        mel = preprocess_for_cnn(audio)
        
        return emotion, mel
    except Exception as e:
        print(f"SVM prediction error: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", np.zeros((128, 128))

# ---- GUI ----
class EmotionGUI:
    def __init__(self, master):
        self.master = master
        master.title("Real-time Speech Emotion Recognition")
        master.geometry("650x600")
        master.configure(bg='#f0f0f0')

        # Title
        title_label = tk.Label(master, text="Speech Emotion Recognition", 
                               font=("Helvetica", 18, "bold"), 
                               bg='#f0f0f0', fg='#2c3e50')
        title_label.pack(pady=10)

        # Model selection frame
        model_frame = tk.Frame(master, bg='#f0f0f0')
        model_frame.pack(pady=10)
        
        self.model_label = tk.Label(model_frame, text="Select Model:", 
                                     font=("Helvetica", 12), 
                                     bg='#f0f0f0')
        self.model_label.pack(side=tk.LEFT, padx=5)
        
        self.model_var = tk.StringVar(value="CNN")
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, 
                                        values=["CNN", "KNN", "SVM"], 
                                        state="readonly", width=15, 
                                        font=("Helvetica", 11))
        self.model_combo.pack(side=tk.LEFT, padx=5)

        # Model status
        self.status_label = tk.Label(master, text="", font=("Helvetica", 9), 
                                     fg="gray", bg='#f0f0f0')
        self.status_label.pack()
        self.update_model_status()

        # Prediction display
        pred_frame = tk.Frame(master, bg='white', relief=tk.RIDGE, borderwidth=2)
        pred_frame.pack(pady=15, padx=20, fill=tk.X)
        
        self.pred_label = tk.Label(pred_frame, text="Detected Emotion: None", 
                                   font=("Helvetica", 16, "bold"), 
                                   fg="#3498db", bg='white')
        self.pred_label.pack(pady=15)

        # Spectrogram canvas
        spec_frame = tk.Frame(master, bg='#f0f0f0')
        spec_frame.pack(pady=10)
        
        self.spec_label = tk.Label(spec_frame, bg='white')
        self.spec_label.pack()

        # Start/Stop buttons
        button_frame = tk.Frame(master, bg='#f0f0f0')
        button_frame.pack(pady=15)
        
        self.start_btn = tk.Button(button_frame, text="▶ Start Listening", 
                                   command=self.start, bg="#27ae60", fg="white",
                                   width=18, height=2, font=("Helvetica", 11, "bold"),
                                   cursor="hand2")
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(button_frame, text="⏹ Stop Listening", 
                                  command=self.stop, bg="#e74c3c", fg="white",
                                  width=18, height=2, font=("Helvetica", 11, "bold"),
                                  state=tk.DISABLED, cursor="hand2")
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Recording indicator
        self.recording_label = tk.Label(master, text="", font=("Helvetica", 10), 
                                       bg='#f0f0f0', fg='#e74c3c')
        self.recording_label.pack()

        self.running = False

    def update_model_status(self):
        """Update status label showing which models are loaded"""
        status = []
        if cnn_model is not None: status.append("CNN ✓")
        if knn_model is not None: status.append("KNN ✓")
        if svm_model is not None: status.append("SVM ✓")
        
        if status:
            self.status_label.config(text=f"Loaded models: {', '.join(status)}")
        else:
            self.status_label.config(text="⚠ No models loaded", fg='red')

    def update_display(self, emotion, mel):
        """Update the GUI with prediction and spectrogram"""
        # Update emotion label
        color_map = {
            "anger": "#e74c3c",
            "happy": "#f39c12",
            "neutral": "#3498db"
        }
        color = color_map.get(emotion.lower(), "#2c3e50")
        self.pred_label.config(text=f"Detected Emotion: {emotion.upper()}", fg=color)

        # Convert spectrogram to image
        fig, ax = plt.subplots(figsize=(7, 3.5))
        img_plot = ax.imshow(mel, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f"Mel Spectrogram - {self.model_var.get()} Model", 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel("Time Frames", fontsize=10)
        ax.set_ylabel("Mel Frequency Bins", fontsize=10)
        plt.colorbar(img_plot, ax=ax, label='dB')
        
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=80)
        plt.close(fig)
        buf.seek(0)
        
        img = Image.open(buf)
        img = img.resize((550, 280))
        photo = ImageTk.PhotoImage(img)
        self.spec_label.config(image=photo)
        self.spec_label.image = photo

    def listen_loop(self):
        """Main recording and prediction loop"""
        if not self.running:
            return
        
        try:
            # Record audio
            print(f"\n{'='*50}")
            print(f"Recording {CHUNK_DURATION} seconds...")
            audio = sd.rec(int(CHUNK_DURATION * SAMPLE_RATE), 
                          samplerate=SAMPLE_RATE, 
                          channels=1, 
                          dtype="float32")
            sd.wait()
            audio = audio.flatten()
            
            # Resample if needed (mic might be 44100 Hz but we want 16000 Hz)
            # Note: This is just a diagnostic - actual resampling handled by sounddevice
            
            # Diagnostic: Check audio quality
            audio_min = np.min(audio)
            audio_max = np.max(audio)
            audio_mean = np.mean(np.abs(audio))
            audio_std = np.std(audio)
            
            print(f"Audio captured:")
            print(f"  - Length: {len(audio)} samples ({len(audio)/SAMPLE_RATE:.2f} seconds)")
            print(f"  - Range: [{audio_min:.4f}, {audio_max:.4f}]")
            print(f"  - Mean absolute: {audio_mean:.4f}")
            print(f"  - Std dev: {audio_std:.4f}")
            print(f"  - Energy: {np.sum(audio**2):.4f}")
            
            # Check if audio is silent
            if audio_max < 0.001:
                print("WARNING: Audio appears to be silent! Check microphone.")
                self.pred_label.config(text="WARNING: No audio detected!", fg='orange')
                if self.running:
                    self.master.after(100, self.listen_loop)
                return
            
            # Check if audio is mostly noise (very low energy with low variance)
            if audio_std < 0.005 and audio_mean < 0.005:
                print("INFO: Audio signal is weak - try speaking louder or closer to mic")
            
            # Get prediction based on selected model
            model_choice = self.model_var.get()
            print(f"\nPredicting with {model_choice}...")
            
            if model_choice == "CNN":
                emotion, mel = predict_cnn(audio)
            elif model_choice == "KNN":
                emotion, mel = predict_knn(audio)
            elif model_choice == "SVM":
                emotion, mel = predict_svm(audio)
            else:
                emotion = "Unknown model"
                mel = np.zeros((128, 128))
            
            print(f"Final prediction: {emotion.upper()}")
            print(f"{'='*50}\n")
            self.update_display(emotion, mel)
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            self.pred_label.config(text=f"Error: {str(e)}", fg='red')
        
        # Schedule next iteration
        if self.running:
            self.master.after(100, self.listen_loop)

    def start(self):
        """Start recording and prediction"""
        if not self.running:
            self.running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.model_combo.config(state=tk.DISABLED)
            self.recording_label.config(text="Recording...")
            print("\n=== Started listening ===")
            self.listen_loop()

    def stop(self):
        """Stop recording and prediction"""
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.model_combo.config(state="readonly")
        self.recording_label.config(text="")
        print("=== Stopped listening ===\n")


if __name__ == "__main__":
    root = tk.Tk()
    gui = EmotionGUI(root)
    root.mainloop()