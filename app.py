"""Streamlit deployment of the trained cell-image research model."""
import io
from pathlib import Path
import threading

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st
import torch

from model import load_model

ROOT = Path(__file__).resolve().parent
MAX_BYTES = 10 * 1024 * 1024


def preprocess(data):
    if not 0 < len(data) <= MAX_BYTES:
        raise ValueError("Choose an image smaller than 10 MB.")
    with Image.open(io.BytesIO(data)) as original:
        if original.format not in {"PNG", "JPEG", "BMP", "TIFF"}:
            raise ValueError("Choose a PNG, JPEG, BMP or TIFF image.")
        if min(original.size) < 16:
            raise ValueError("The image must be at least 16 by 16 pixels.")
        if original.width * original.height > 20_000_000:
            raise ValueError("Choose an image with fewer than 20 million pixels.")
        image = ImageOps.exif_transpose(original).convert("RGB")
        preview = image.copy()
        preview.thumbnail((640, 640))
        image = image.resize((128, 128), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0), preview


@st.cache_resource
def classifier():
    torch.set_num_threads(2)
    model, saved = load_model(ROOT / "weights" / "best.pt")
    return model, float(saved["threshold"]), threading.Lock()


st.set_page_config(page_title="Malaria cell classifier", page_icon="🔬", layout="centered")
st.title("Malaria cell classifier")
st.write("Explore a trained CNN–LSTM using a cropped red blood cell image.")
st.info("Research demonstration only. The model classifies individual cell images; "
        "it does not diagnose a patient's malaria status. Its score is not a calibrated clinical probability.")

source = st.radio("Image source", ["Upload a cell image", "Try a sample"], horizontal=True)
data = None
if source == "Upload a cell image":
    uploaded = st.file_uploader("Choose a cropped cell image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    st.caption("Images are sent to the hosting server for processing. This app does not save them "
               "or keep a shared prediction history. Please use images without patient identifiers.")
    if uploaded is not None:
        data = uploaded.getvalue()
else:
    sample = st.selectbox("Bundled sample", ["Sample 1", "Sample 2", "Sample 3", "Sample 4"])
    data = (ROOT / "samples" / f"sample_{sample[-1]}.png").read_bytes()
    st.caption("These four examples were selected to show two correct predictions and two errors. "
               "They are not an accuracy estimate.")

if data is not None:
    try:
        tensor, preview = preprocess(data)
        st.image(preview, caption="Selected cell image", width=320)
        if st.button("Run prediction", type="primary"):
            with st.spinner("Analyzing the cell…"):
                model, threshold, lock = classifier()
                with lock, torch.inference_mode():
                    score = model(tensor).sigmoid().item()
            label = "Parasitized" if score >= threshold else "Uninfected"
            st.subheader(f"Model prediction: {label}")
            st.metric("Parasitized model score", f"{score:.2%}")
            st.caption(f"Decision threshold: {threshold:.2f}. A score at or above the threshold is classified as Parasitized.")
    except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError):
        st.error("Could not read this image. Use a valid cropped cell image between 16 × 16 pixels "
                 "and 20 million pixels, smaller than 10 MB.")
    except Exception:
        st.error("Prediction is temporarily unavailable. Please try again later.")
