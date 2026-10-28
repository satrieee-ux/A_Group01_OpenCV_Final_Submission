"""Phase 2 Streamlit app (styled): camera/upload -> 4 categories + handling guidance.
 
Run from the project root:  py -m streamlit run app_v2.py
Optional: put your banner image at assets/banner.png (used as the top banner).
"""
import cv2
import streamlit as st
 
from src import ui_v2 as ui
from src.config import PROJECT_ROOT
from src.guidance_v2 import (
    B3_NOT_RECOGNIZED, B3_NOT_RECOGNIZED_INTRO, FOOTER, SOURCES, guidance_for,
)
from src.predict_v2 import predict_bgr_v2
from src.preprocessing import InvalidImageError, decode_image_bytes
from src.taxonomy import CATEGORIES, CATEGORY_NAMES
from src.v2_config import B3_CAUTION_THRESHOLD, CATEGORY_REJECT_THRESHOLD, MODEL_V2_PATH
 
MAX_UPLOAD_MB = 10
BANNER = PROJECT_ROOT / "assets" / "banner.png"
 
st.set_page_config(page_title="Asisten Pemilahan Sampah", page_icon="♻️", layout="centered")
st.markdown(ui.CSS, unsafe_allow_html=True)
 
 
def show(html_str: str):
    st.markdown(html_str, unsafe_allow_html=True)
 
 
def render_guidance(cat: str, manual: bool = False):
    g = guidance_for(cat)
    show(ui.guidance_card(cat, g["title"], g["summary"], [t for t, _ in g["steps"]], manual))
    tags = sorted({t.strip() for _, tag in g["steps"] for t in tag.split(",")})
    with st.expander("Sumber panduan"):
        for t in tags:
            if t == "UMUM":
                st.write("UMUM: saran umum tanpa sumber (belum diverifikasi).")
            else:
                st.write(f"{t}: {SOURCES.get(t, t)}")
    st.caption(FOOTER)
 
 
# ---------- header ----------
show(ui.hero_blue(BANNER))
show(ui.hero_chips())
st.caption(
    "Alat bantu edukasi. Foto atau unggah gambar satu barang; model memperkirakan "
    "kategorinya dan menampilkan panduan penanganan umum. Model hanya mengenal 11 jenis "
    "barang dan bisa salah."
)
 
if not MODEL_V2_PATH.exists():
    show(ui.err_box("Model tidak ditemukan",
                    f"{MODEL_V2_PATH}. Jalankan training v2 terlebih dahulu."))
    st.stop()
 
# ---------- input source (two tiles) ----------
if "mode" not in st.session_state:
    st.session_state["mode"] = "camera"
show(ui.section("Pilih sumber gambar"))
col_cam, col_up = st.columns(2)
if col_cam.button("📷  Kamera", key="mode_cam",
                  type="primary" if st.session_state["mode"] == "camera" else "secondary"):
    st.session_state["mode"] = "camera"
    st.rerun()
if col_up.button("🖼️  Unggah file", key="mode_up",
                 type="primary" if st.session_state["mode"] == "upload" else "secondary"):
    st.session_state["mode"] = "upload"
    st.rerun()
 
if st.session_state["mode"] == "camera":
    shot = st.camera_input("Ambil foto satu barang", label_visibility="collapsed")
    if shot is None:
        st.info("Belum ada foto. Jika kamera tidak muncul, izinkan akses kamera di browser "
                "dan buka aplikasi lewat localhost.")
else:
    shot = st.file_uploader("Pilih gambar (JPG atau PNG)", type=["jpg", "jpeg", "png"],
                            label_visibility="collapsed")
    if shot is None:
        st.info("Belum ada gambar yang dipilih.")
 
with st.expander("B3 lain yang tidak dikenali model"):
    st.write(B3_NOT_RECOGNIZED_INTRO)
    for text, _tag in B3_NOT_RECOGNIZED:
        st.markdown(f"- {text}")
 
if shot is None:
    st.stop()
 
# ---------- validation ----------
data = shot.getvalue()
if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
    show(ui.err_box("File terlalu besar", f"Ukuran file lebih dari {MAX_UPLOAD_MB} MB. "
                    "Pilih gambar yang lebih kecil."))
    st.stop()
try:
    img_bgr = decode_image_bytes(data)
except InvalidImageError as e:
    show(ui.err_box("Gambar tidak valid", f"{e} Silakan pilih file lain."))
    st.stop()
 
st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), width=360)
 
# ---------- prediction ----------
try:
    with st.spinner("Mengklasifikasi ..."):
        pred = predict_bgr_v2(img_bgr)
except Exception as e:  # shown to the user, never replaced by a made-up prediction
    show(ui.err_box("Klasifikasi gagal", f"{type(e).__name__}: {e}"))
    st.stop()
 
show(ui.section("Hasil"))
p_b3 = pred.category_probs["b3"]
if pred.accepted:
    show(ui.result_card(pred.category, pred.item, pred.category_confidence))
    if pred.b3_caution and pred.category != "b3":
        show(ui.err_box("PERINGATAN B3", f"Ada kemungkinan {p_b3:.0%} bahwa ini sampah B3. "
                        "Jika ragu, perlakukan sebagai B3."))
    render_guidance(pred.category)
else:
    show(ui.err_box("Prediksi belum andal", pred.message))
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
    show(ui.bars(pred.category_probs))
    st.write("Skor per jenis barang:")
    for item, p in sorted(pred.item_probs.items(), key=lambda kv: -kv[1]):
        st.write(f"{item}: {p:.1%}")
    st.write(f"Ambang penolakan: {CATEGORY_REJECT_THRESHOLD:.0%} | "
             f"ambang peringatan B3: {B3_CAUTION_THRESHOLD:.0%}")
 
st.caption("Keyakinan model bukan peluang jawabannya benar, dan penolakan berdasarkan "
           "keyakinan tidak menjamin barang di luar jenis yang dikenali selalu ditolak.")