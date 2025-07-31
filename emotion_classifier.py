# emotion_classifier.py

import os
import argparse

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from utils import extract_features

# Map RAVDESS filename codes to emotion labels
EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',      # not included
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',   # not included
    '07': 'disgust',   # not included
    '08': 'surprised'  # not included
}

# Let's just stick to these four for now
TARGET_EMOTIONS = {'neutral', 'happy', 'sad', 'angry'}


def load_data(data_dir):
    """
    Walk data_dir looking for .wav files, extract features, and return
    X (DataFrame) and y (Series) containing features and labels.
    """
    records = []
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if not fname.lower().endswith('.wav'):
                continue

            parts = fname.split('-')
            if len(parts) < 3:
                continue  # skip unexpected filenames

            code = parts[2]
            label = EMOTION_MAP.get(code)
            if label not in TARGET_EMOTIONS:
                continue

            path = os.path.join(root, fname)
            try:
                feats = extract_features(path)
            except Exception as e:
                print(f"⚠️  Failed to extract features from {fname}: {e}")
                continue

            records.append((feats, label))

    if not records:
        raise RuntimeError(f"No valid .wav files found under {data_dir}")

    X = np.stack([r[0] for r in records], axis=0)
    y = [r[1] for r in records]
    return pd.DataFrame(X), pd.Series(y)


def train_and_save_model(data_dir, model_path):
    """
    Train multiple classifiers, select the best by 5-fold CV, retrain on full
    split, evaluate on hold-out, and save the model to disk.
    """
    print(f" Loading data from '{data_dir}'...")
    X, y = load_data(data_dir)
    print(f" Loaded {len(y)} samples across {len(set(y))} emotions.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(y_train)} samples; Test: {len(y_test)} samples")

    candidates = {
        'SVM (linear)': SVC(kernel='linear', probability=True),
        'Logistic Regression': LogisticRegression(max_iter=300)
    }

    best_score = -np.inf
    best_name = None
    best_model = None

    for name, model in candidates.items():
        print(f" CV on {name}...")
        scores = cross_val_score(model, X_train, y_train, cv=5)
        mean_acc = scores.mean()
        print(f"   • CV accuracy: {mean_acc:.3f}")
        if mean_acc > best_score:
            best_score, best_name, best_model = mean_acc, name, model

    print(f"\n Best model: {best_name} (CV acc: {best_score:.3f})")
    best_model.fit(X_train, y_train)
    test_acc = best_model.score(X_test, y_test)
    print(f" Test set accuracy: {test_acc:.3f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    dump(best_model, model_path)
    print(f" Model saved to: {model_path}")


def predict_emotion(file_path, model_path="models/emotion_model.pkl"):
    """
    Load the trained model from model_path, extract features from file_path,
    and return the predicted emotion label (one of: happy, sad, neutral, angry).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'")

    model = load(model_path)
    feats = extract_features(file_path).reshape(1, -1)
    prediction = model.predict(feats)[0]
    return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or test the emotion classifier."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train on data_dir and save the best model to model_path"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/ravdess",
        help="Directory containing emotion-labeled WAV files"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/emotion_model.pkl",
        help="Where to save (or load) the classifier"
    )
    args = parser.parse_args()

    if args.train:
        train_and_save_model(args.data_dir, args.model_path)
    else:
        parser.print_help()
