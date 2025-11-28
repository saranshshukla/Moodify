from pathlib import Path
import numpy as np
import librosa

def extract_mfcc_features(file_path: str, n_mfcc: int = 13, sr: int | None = None) -> np.ndarray:
    """
    Load audio and return a simple feature vector: MFCCs + delta, averaged over time.
    Keep this consistent with training.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Audio not found: {file_path}")

    y, sampling_rate = librosa.load(str(p), sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sampling_rate, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)

    stacked = np.vstack([mfcc, delta])
    feat = stacked.mean(axis=1)   # shape (2 * n_mfcc,)
    return feat
