"""Phase 2 Streamlit app: camera/upload -> category (4) + handling guidance.

Run from the project root:  py -m streamlit run app_v2.py
"""
import cv2
import streamlit as st

from src.guidance_v2 import (
    B3_NOT_RECOGNIZED, B3_NOT_RECOGNIZED_INTRO, FOOTER, SOURCES, guidance_for,
)
from src.predict_v2 import predict_bgr_v2
from src.preprocessing import InvalidImageError, decode_image_bytes
from src.taxonomy import CATEGORIES, CATEGORY_NAMES
from src.v2_config import B3_CAUTION_THRESHOLD, CATEGORY_REJECT_THRESHOLD, MODEL_V2_PATH

MAX_UPLOAD_MB = 10

st.set_page_config(page_title="Asisten Pemilahan Sampah", page_icon="♻️")
st.title("♻️ Asisten Klasifikasi dan Pemilahan Sampah")
st.write(
    "Foto atau unggah gambar **satu** barang. Aplikasi memperkirakan kategorinya "
    "(**organik, anorganik, B3, residu**) dan menampilkan panduan penanganan umum."
)
st.caption(
    "Alat bantu edukasi. Model hanya mengenal 11 jenis barang dan bisa salah. "
    "Keyakinan model bukan peluang jawabannya benar."
)

if not MODEL_V2_PATH.exists():
    st.error(f"File model tidak ditemukan: {MODEL_V2_PATH}. Jalankan training v2 terlebih dahulu.")
    st.stop()


def render_guidance(cat: str, manual: bool = False):
    g = guidance_for(cat)
    if manual:
        st.caption("Panduan berdasarkan pilihan Anda, bukan prediksi model.")
    st.markdown(f"**{g['title']}**: {g['summary']}")
    for i, (text, _tag) in enumerate(g["steps"], 1):
        st.markdown(f"{i}. {text}")
    tags = sorted({t.strip() for _, tag in g["steps"] for t in tag.split(",")})
    with st.expander("Sumber panduan"):
        for t in tags:
            if t == "UMUM":
                st.write("UMUM: saran umum tanpa sumber (belum diverifikasi).")
            else:
                st.write(f"{t}: {SOURCES.get(t, t)}")
    st.caption(FOOTER)


mode = st.radio("Sumber gambar", ["Kamera", "Unggah file"], horizontal=True)
if mode == "Kamera":
    shot = st.camera_input("Arahkan kamera ke satu barang, lalu ambil foto")
    if shot is None:
        st.info("Belum ada foto. Jika kamera tidak muncul, izinkan akses kamera di browser.")
else:
    shot = st.file_uploader("Pilih gambar (JPG atau PNG)", type=["jpg", "jpeg", "png"])
    if shot is None:
        st.info("Belum ada gambar yang dipilih.")

with st.expander("B3 lain yang tidak dikenali model"):
    st.write(B3_NOT_RECOGNIZED_INTRO)
    for text, _tag in B3_NOT_RECOGNIZED:
        st.markdown(f"- {text}")

if shot is None:
    st.stop()

data = shot.getvalue()
if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
    st.error(f"Ukuran file lebih dari {MAX_UPLOAD_MB} MB. Pilih gambar yang lebih kecil.")
    st.stop()
try:
    img_bgr = decode_image_bytes(data)
except InvalidImageError as e:
    st.error(f"Gambar tidak valid: {e} Silakan pilih file lain.")
    st.stop()

st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), width=400)

try:
    with st.spinner("Mengklasifikasi ..."):
        pred = predict_bgr_v2(img_bgr)
except Exception as e:  # shown to the user, never replaced by a made-up prediction
    st.error(f"Klasifikasi gagal: {type(e).__name__}: {e}")
    st.stop()

st.subheader("Hasil")
p_b3 = pred.category_probs["b3"]
if pred.accepted:
    box = st.error if pred.category == "b3" else st.success
    box(f"Kategori: **{CATEGORY_NAMES[pred.category]}**")
    st.write(f"Keyakinan model: **{pred.category_confidence:.1%}** | "
             f"perkiraan jenis barang: **{pred.item}**")
    if pred.b3_caution and pred.category != "b3":
        st.error(f"PERINGATAN: ada kemungkinan {p_b3:.0%} bahwa ini sampah B3. "
                 "Jika ragu, perlakukan sebagai B3.")
    render_guidance(pred.category)
else:
    st.warning("Prediksi belum cukup andal untuk ditampilkan sebagai hasil.")
    st.write(pred.message)
    st.caption(f"Tebakan terdekat (keyakinan rendah, bukan hasil): "
               f"{CATEGORY_NAMES[pred.category]} ({pred.category_confidence:.1%}).")
    choice = st.selectbox(
        "Pilih kategori secara manual untuk melihat panduannya", CATEGORIES,
        index=None, placeholder="Pilih kategori",
        format_func=lambda c: CATEGORY_NAMES[c])
    if choice:
        render_guidance(choice, manual=True)

with st.expander("Rincian skor"):
    st.write("Skor per kategori (jumlah skor jenis barang di dalamnya):")
    for c in CATEGORIES:
        st.write(f"{CATEGORY_NAMES[c]}: {pred.category_probs[c]:.1%}")
        st.progress(pred.category_probs[c])
    st.write("Skor per jenis barang:")
    for item, p in sorted(pred.item_probs.items(), key=lambda kv: -kv[1]):
        st.write(f"{item}: {p:.1%}")
    st.write(f"Ambang penolakan: {CATEGORY_REJECT_THRESHOLD:.0%} | "
             f"ambang peringatan B3: {B3_CAUTION_THRESHOLD:.0%}")