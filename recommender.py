# recommender.py

import json
import os

def get_recommendations(emotion: str, songs_json_path: str = "data/songs.json"):
    if not os.path.exists(songs_json_path):
        raise FileNotFoundError(f"Could not find '{songs_json_path}'")

    with open(songs_json_path, "r", encoding="utf-8") as f:
        song_map = json.load(f)

    # Fallback to neutral if missing
    return song_map.get(emotion, song_map.get("neutral", []))
