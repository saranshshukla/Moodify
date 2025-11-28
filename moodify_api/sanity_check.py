# sanity_check.py
from api_core.audio_processing import extract_mfcc_features
from api_core.model_loader import load_model
import numpy as np
print("Extracting features from a sample (replace with a real sample path)")
feat = extract_mfcc_features("/home/dvoid/Documents/Moodify/data/ravdess/Actor_01/03-02-01-01-01-01-01.wav")  # point to a known small wav
print("feature shape:", feat.shape)
model = load_model()   # uses default lookups
print("model n_features_in_:", getattr(model, "n_features_in_", "unknown"))
