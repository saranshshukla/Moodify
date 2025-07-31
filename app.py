import streamlit as st
import os
from emotion_classifier import predict_emotion
from recommender import get_recommendations

# — Page Configuration —
st.set_page_config(
    page_title="Moodify • Emotion-Aware Music Recommender",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# — Custom Styling: dark theme + useful white upload box —
st.markdown("""
    <style>
        /* Overall page */
        body, .stApp {
            background-color: #121212;
            color: #E0E0E0;
        }
        /* Headings */
        h1, h2, h3, h4 {
            color: #FFFFFF;
            text-align: center;
            margin: 0.5rem 0;
        }
        /* Upload box */
        .upload-area {
            background-color: #FFFFFF;
            padding: 2rem;
            margin: 1rem auto;
            width: 80%;
            max-width: 500px;
            border: 2px dashed #888888;
            border-radius: 10px;
            text-align: center;
        }
        .upload-area:hover {
            border-color: #BBBBBB;
            background-color: #FAFAFA;
        }
        .upload-instruction {
            color: #555555;
            font-size: 1rem;
            margin-bottom: 1rem;
        }
        /* Results */
        .result {
            margin-top: 1.5rem;
            font-size: 1.3rem;
            text-align: center;
        }
        .result.success strong {
            color: #4caf50;
        }
        .result.warning strong {
            color: #ff9800;
        }
        /* Recommendation boxes */
        .song-box {
            background-color: #1E1E1E;
            border-left: 4px solid #3F51B5;
            padding: 1rem;
            margin: 0.75rem auto;
            width: 80%;
            max-width: 500px;
            border-radius: 6px;
            color: #E0E0E0;
        }
        .song-box a {
            color: #64B5F6;
            text-decoration: none;
        }
    </style>
""", unsafe_allow_html=True)

# — Title —
st.markdown("<h1>Moodify</h1>", unsafe_allow_html=True)
st.markdown("<h3>Let your voice choose the soundtrack</h3>", unsafe_allow_html=True)

# — Upload Section —
st.markdown("<div class='upload-area'>", unsafe_allow_html=True)
st.markdown(
    "<div class='upload-instruction'>Drag & drop your .wav file here<br>or click to browse</div>",
    unsafe_allow_html=True
)
uploaded_file = st.file_uploader("", type=["wav"], label_visibility="collapsed")
st.markdown("</div>", unsafe_allow_html=True)

# — Emotion Detection & Recommendations —
if uploaded_file:
    # Save temp file
    temp_path = "temp_input.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.markdown("---")
    st.markdown("<div class='result'><strong>Processing your voice…</strong></div>", unsafe_allow_html=True)

    # Predict
    emotion = predict_emotion(temp_path, "models/emotion_model.pkl")
    if emotion.lower() == "unknown":
        st.markdown(
            "<div class='result warning'><strong>"
            "Unable to determine your emotion. Try a clearer recording."
            "</strong></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result success'><strong>"
            f"Detected Emotion: {emotion.capitalize()}"
            "</strong></div>",
            unsafe_allow_html=True
        )
        st.markdown("<h4>Curated Recommendations</h4>", unsafe_allow_html=True)

        # Fetch recommendations
        recs = get_recommendations(emotion)

        # Render each song as a clickable title
        for song in recs:
            title = song.get("title", "Unknown Track")
            link  = song.get("link", "#")
            st.markdown(
                f"<div class='song-box'>"
                f"<a href='{link}' target='_blank'>{title}</a>"
                "</div>",
                unsafe_allow_html=True
            )

    # Clean up
    os.remove(temp_path)
