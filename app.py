"""Stage 6: Streamlit interface.

Run from the project root:  py -m streamlit run app.py
"""
import cv2
import streamlit as st

from src.config import CONFIDENCE_THRESHOLD, TARGET_CLASSES
from src.predict import get_model, predict_bgr
from src.preprocessing import InvalidImageError, decode_image_bytes

MAX_UPLOAD_MB = 10
DISPLAY_NAMES = {
    "plastic_bottle": "Plastic bottle",
    "metal_can": "Metal can",
    "paper": "Paper",
}

st.set_page_config(page_title="Waste Classification Assistant", page_icon="♻️")


@st.cache_resource
def load_model_once():
    return get_model()


st.title("♻️ Waste Classification and Sorting Assistant")
st.write(
    "Upload a photo of **one** waste item. The app predicts whether it is a "
    "**plastic bottle, metal can, or paper** and gives general sorting guidance."
)
st.caption(
    "Educational decision-support tool. It only knows these three categories and "
    "can be wrong. Local recycling rules may differ."
)

try:
    load_model_once()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

uploaded = st.file_uploader("Choose an image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("No image selected yet.")
    st.stop()

data = uploaded.getvalue()
if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
    st.error(f"The file is larger than {MAX_UPLOAD_MB} MB. Please choose a smaller image.")
    st.stop()

try:
    img_bgr = decode_image_bytes(data)  # OpenCV decode and validation
except InvalidImageError as e:
    st.error(f"Invalid image: {e} Please upload a different file.")
    st.stop()

st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption=uploaded.name, width=400)

try:
    with st.spinner("Classifying ..."):
        pred = predict_bgr(img_bgr)
except Exception as e:  # shown to the user, never replaced by a made-up prediction
    st.error(f"Classification failed: {type(e).__name__}: {e}")
    st.stop()

st.subheader("Result")
if pred.accepted:
    st.success(f"Predicted category: **{DISPLAY_NAMES[pred.label]}**")
    st.write(f"Model confidence: **{pred.confidence:.1%}**")
    st.info(pred.guidance)
else:
    st.warning("The prediction is not reliable enough to show a category.")
    st.write(pred.message)
    st.write(
        f"Closest category (low confidence, not shown as a result): "
        f"{DISPLAY_NAMES[pred.label]} ({pred.confidence:.1%})"
    )

st.caption(
    "Confidence is the model's own score. It is not the probability that the "
    "answer is correct, and the model can be confident about items it does not "
    "know. The rejection threshold reduces, but does not remove, this risk."
)

with st.expander("Details: score per category"):
    for c in TARGET_CLASSES:
        st.write(f"{DISPLAY_NAMES[c]}: {pred.probabilities[c]:.1%}")
        st.progress(pred.probabilities[c])
    st.write(f"Rejection threshold: {CONFIDENCE_THRESHOLD:.0%}")