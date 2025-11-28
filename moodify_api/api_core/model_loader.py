from pathlib import Path
from joblib import load
from typing import Any
import threading

_model: Any | None = None
_model_lock = threading.Lock()

def _candidate_model_paths() -> list:
    here = Path(__file__).resolve()
    candidates = []

    # repo root (two levels up if structure is repo_root/moodify_api/api_core)
    repo_root = here.parents[2] if len(here.parents) >= 3 else here.parents[0]
    candidates.append(repo_root / "models" / "emotion_model.pkl")

    # moodify_api/models (in case someone put model inside api folder)
    candidates.append(here.parents[1] / "models" / "emotion_model.pkl")

    # current working directory /models
    candidates.append(Path.cwd() / "models" / "emotion_model.pkl")

    # direct fallback in repo root
    candidates.append(repo_root / "emotion_model.pkl")
    return [p for p in candidates if p.exists()]

def load_model(model_path: str | None = None) -> Any:
    """
    Load and cache the sklearn model. Try a few sensible default paths
    if no explicit path is provided.
    """
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model

        if model_path:
            mp = Path(model_path)
            if not mp.exists():
                raise FileNotFoundError(f"Model file not found at '{model_path}'")
            _model = load(str(mp))
            return _model

        found = _candidate_model_paths()
        if not found:
            raise FileNotFoundError("Model file not found. Please put emotion_model.pkl in models/ at repo root.")
        _model = load(str(found[0]))
        return _model

