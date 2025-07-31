import numpy as np
import librosa

def extract_features(file_path: str, n_mfcc: int = 13) -> np.ndarray:
    """
    Load a WAV file, compute MFCCs + deltas, and return their mean vector.
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
    except Exception as e:
        raise RuntimeError(f"Could not load '{file_path}': {e}")

    #  MFCC matrix: (n_mfcc, frames)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    #  First-order deltas (same shape)
    delta = librosa.feature.delta(mfcc)

    # Stack (2*n_mfcc, frames) and average over time → (2*n_mfcc,)
    feat = np.vstack([mfcc, delta])
    return feat.mean(axis=1)
