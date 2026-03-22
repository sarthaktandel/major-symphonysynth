import streamlit as st
import subprocess
from pathlib import Path
import os
import time

st.set_page_config(page_title="SymphonySynth", layout="centered")

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
    <style>
        .main {
            padding-top: 2rem;
        }
        .stButton button {
            background-color: #ff4b4b;
            color: white;
            border-radius: 8px;
            height: 3em;
            width: 100%;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.title("🎼 SymphonySynth")
st.subheader("AI-Based Melody Generator")

st.write(
    "Generate musical melodies using Transformer-based symbolic sequence modeling."
)

st.markdown("---")

# -----------------------------
# Controls
# -----------------------------
st.markdown("### 🎛️ Generation Controls")

col1, col2 = st.columns(2)

with col1:
    temperature = st.slider("Temperature", 0.5, 1.5, 0.9)

with col2:
    top_k = st.slider("Top-K", 5, 50, 20)

# -----------------------------
# Upload
# -----------------------------
st.markdown("### 🎹 Seed Input")

seed_path = None
seed_file = st.file_uploader("Upload Seed MIDI (optional)", type=["mid"])

if seed_file is not None:
    seed_path = Path("temp_seed.mid")
    with open(seed_path, "wb") as f:
        f.write(seed_file.read())
    st.success("✅ Seed uploaded successfully")

st.markdown("---")

# -----------------------------
# Generate
# -----------------------------
if st.button("🚀 Generate Melody"):

    status = st.empty()
    progress = st.progress(0)

    status.info("⏳ Initializing model...")
    time.sleep(0.5)
    progress.progress(10)

    cmd = ["python", "src/inference/generate_melody.py"]

    if seed_path:
        cmd.extend(["--seed", str(seed_path)])

    status.info("📦 Loading model...")
    time.sleep(0.5)
    progress.progress(30)

    status.info("🎼 Processing input...")
    time.sleep(0.5)
    progress.progress(50)

    status.info("🎹 Generating melody...")
    
    subprocess.run(cmd)

    progress.progress(80)
    status.info("🧠 Finalizing output...")
    time.sleep(0.5)

    output_path = Path("outputs/generated.mid")

    if output_path.exists():
        progress.progress(100)
        status.success("🎉 Melody Generated Successfully!")

        st.markdown("### 📥 Output")
        st.info("Download the MIDI file and play it in VLC / MuseScore / GarageBand")

        with open(output_path, "rb") as f:
            st.download_button(
                label="⬇ Download MIDI",
                data=f,
                file_name="generated.mid"
            )
    else:
        status.error("❌ Generation failed")

# -----------------------------
# Cleanup
# -----------------------------
if seed_path and seed_path.exists():
    os.remove(seed_path)