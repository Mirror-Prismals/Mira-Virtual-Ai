import os
import numpy as np
import librosa
import tensorflow as tf
import keras
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import argparse

# ==========================
# Configuration Parameters
# ==========================
SAMPLE_RATE = 22050       # Sample rate for audio files
SNIPPET_DURATION = 30     # Duration of each snippet in seconds
N_MFCC = 13               # Number of MFCCs to extract
HOP_LENGTH = 512
N_FFT = 2048
BATCH_SIZE = 32
EPOCHS = 50
MODEL_SAVE_PATH = 'genre_model.h5'
LABEL_ENCODER_PATH = 'label_encoder.npy'

# ==========================
# Data Preparation
# ==========================
def extract_snippets(file_path, sr=SAMPLE_RATE, duration=SNIPPET_DURATION):
    """
    Extracts snippets from a full-length audio file.
    """
    try:
        signal, sr = librosa.load(file_path, sr=sr)
        total_duration = librosa.get_duration(y=signal, sr=sr)
        snippets = []
        
        if total_duration < duration:
            # Pad the signal if it's shorter than the snippet duration
            pad_length = int(sr * duration) - len(signal)
            signal = np.pad(signal, (0, pad_length), 'constant')
            snippets.append(signal)
        else:
            # Extract multiple snippets evenly spaced
            num_snippets = total_duration // duration
            for i in range(int(num_snippets)):
                start_sample = i * int(sr * duration)
                end_sample = start_sample + int(sr * duration)
                snippet = signal[start_sample:end_sample]
                snippets.append(snippet)
        return snippets
    except Exception as e:
        print(f"Error extracting snippets from {file_path}: {e}")
        return []

def extract_features(snippets, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    """
    Extracts MFCC features from a list of audio snippets.
    """
    features = []
    for snippet in snippets:
        mfcc = librosa.feature.mfcc(y=snippet, sr=sr, n_mfcc=n_mfcc, hop_length=HOP_LENGTH, n_fft=N_FFT)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        features.append(mfcc_mean)
    return features

def load_dataset(dataset_path):
    """
    Loads the dataset, extracts features and labels.
    Assumes each subdirectory in dataset_path is a genre label.
    """
    X = []
    y = []
    genres = os.listdir(dataset_path)
    print(f"Found genres: {genres}")
    
    for genre in genres:
        genre_path = os.path.join(dataset_path, genre)
        if not os.path.isdir(genre_path):
            continue
        for file in os.listdir(genre_path):
            if file.lower().endswith(('.wav', '.mp3', '.flac')):
                file_path = os.path.join(genre_path, file)
                snippets = extract_snippets(file_path)
                features = extract_features(snippets)
                X.extend(features)
                y.extend([genre] * len(features))
    return np.array(X), np.array(y)

# ==========================
# Model Building
# ==========================
def build_model(input_shape, num_classes):
    """
    Builds and compiles the DNN model.
    """
    model = models.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_shape,)),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ==========================
# Training
# ==========================
def train_model(dataset_path):
    """
    Trains the DNN model on the provided dataset.
    """
    print("Loading dataset...")
    X, y = load_dataset(dataset_path)
    print(f"Total samples: {len(X)}")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    print(f"Number of classes: {num_classes}")
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, 
                                                        test_size=0.2, 
                                                        random_state=42, 
                                                        stratify=y_encoded)
    
    # Build model
    model = build_model(X_train.shape[1], num_classes)
    model.summary()
    
    # Train
    print("Training model...")
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE)
    
    # Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print(classification_report(y_test, y_pred_classes, target_names=le.classes_))
    
    # Save model and label encoder
    model.save(MODEL_SAVE_PATH)
    np.save(LABEL_ENCODER_PATH, le.classes_)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print(f"Label encoder saved to {LABEL_ENCODER_PATH}")

# ==========================
# Inference
# ==========================
def describe_audio(file_path):
    """
    Describes the genre of the provided audio file using the trained model.
    """
    if not os.path.exists(MODEL_SAVE_PATH) or not os.path.exists(LABEL_ENCODER_PATH):
        return "Model or label encoder not found. Please train the model first."
    
    # Load model and label encoder
    model = keras.models.load_model(MODEL_SAVE_PATH)
    genres = np.load(LABEL_ENCODER_PATH, allow_pickle=True)
    
    try:
        # Extract snippets and features
        snippets = extract_snippets(file_path)
        if not snippets:
            return "No valid snippets extracted from the audio file."
        features = extract_features(snippets)
        X = np.array(features)
        
        # Predict for each snippet and take the most common prediction
        predictions = model.predict(X)
        predicted_indices = np.argmax(predictions, axis=1)
        predicted_genres = genres[predicted_indices]
        
        # Determine the most frequent genre in predictions
        unique, counts = np.unique(predicted_genres, return_counts=True)
        genre_counts = dict(zip(unique, counts))
        top_genre = max(genre_counts, key=genre_counts.get)
        
        description = f"Genre: {top_genre.capitalize()}\n"
        description += "Key: Detection not implemented.\n"
        description += "Instruments: Detection not implemented."
        return description
    except Exception as e:
        return f"Error processing file: {e}"

# ==========================
# Main Function
# ==========================
def main():
    parser = argparse.ArgumentParser(description="Music Description Tool")
    parser.add_argument('mode', choices=['train', 'describe'], 
                        help="Mode: 'train' to train the model or 'describe' to describe an audio file.")
    parser.add_argument('--dataset_path', type=str, help="Path to the dataset folder (required for training).")
    parser.add_argument('--file', type=str, help="Path to the audio file for description (required for describe).")
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        if not args.dataset_path:
            print("Please provide the dataset path using --dataset_path")
            return
        if not os.path.exists(args.dataset_path):
            print(f"Dataset path {args.dataset_path} does not exist.")
            return
        train_model(args.dataset_path)
    elif args.mode == 'describe':
        if not args.file:
            print("Please provide the audio file path using --file")
            return
        if not os.path.exists(args.file):
            print(f"Audio file {args.file} does not exist.")
            return
        description = describe_audio(args.file)
        print(description)

if __name__ == "__main__":
    main()
