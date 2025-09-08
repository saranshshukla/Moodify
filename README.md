# 🎵 Moodify

A simple, Python-powered **music recommender** that listens to your voice and instantly picks tracks to match your mood.  

Moodify uses classic **machine learning** (no heavy deep-learning frameworks here) to turn your `.wav` voice clip into an **emotional fingerprint** — happy, sad, neutral, or angry — and then serves up a few hand-curated Spotify tracks or playlists to suit how you feel.  

---

## ✨ Key Features

- **🎙️ Audio Processing** → Quickly extracts MFCCs (and deltas) from any WAV file.  
- **🧠 Emotion Classification** → Transparent SVM or Logistic Regression model — interpretable and reliable.  
- **🎶 Music Recommendations** → Static mapping links each mood to a concise, thoughtfully chosen set of songs or playlists.  
- **💻 Friendly UI** → Custom styled Streamlit app with drag-and-drop upload, dark/light toggle, and clickable song titles.  
- **⚡ Easy Setup & Deployment** → Train in one command, launch with `streamlit run app.py`, and you’re live (perfect for Streamlit Cloud).  

---

## 🛠️ Tech Stack
- **Python**  
- **librosa** (audio processing)  
- **numpy** (data manipulation)  
- **scikit-learn** (ML models)  
- **joblib** (model persistence)  
- **pandas** (data handling)  
- **Streamlit** (web UI)  

---

## 🚀 Getting Started

Follow these steps to set up and run Moodify locally:

1. **Clone this repo**
   ```bash
   git clone https://github.com/yourusername/moodify.git
   cd moodify
