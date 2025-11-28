from typing import Optional
from .audio_processing import extract_mfcc_features
from .model_loader import load_model

def predict_from_file(file_path: str, model_path: Optional[str] = None) -> str:
    """
    Given a local audio file path, return the predicted emotion label.
    """
    feat = extract_mfcc_features(file_path)    # shape (2 * n_mfcc,)
    X = feat.reshape(1, -1)                    # sklearn expects 2D input
    model = load_model(model_path)
    pred = model.predict(X)[0]
    return str(pred)