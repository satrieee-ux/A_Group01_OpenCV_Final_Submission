"""Pure HTML/CSS builders for the v2 Streamlit app (no Streamlit import here).
 
Colours of the four categories follow the colour identities in
Instruksi Gubernur DKI Jakarta No. 5/2026, Lampiran A:
organik = hijau, anorganik = kuning, B3 = merah, residu = abu-abu.
The look is inspired by mobile-app card layouts (rounded cards, soft shadows,
colour-coded badges). No third-party logos or brand assets are used.
"""
import base64
import html
from pathlib import Path
 
CAT_STYLE = {
    "organik":   {"color": "#22A06B", "bg": "#E8F7F0", "text": "#166A49", "icon": "🍃", "label": "Organik"},
    "anorganik": {"color": "#F5B700", "bg": "#FFF6DB", "text": "#8A6100", "icon": "♻️", "label": "Anorganik"},
    "b3":        {"color": "#E5484D", "bg": "#FDECEC", "text": "#B42318", "icon": "⚠️", "label": "B3"},
    "residu":    {"color": "#6B7280", "bg": "#F0F2F5", "text": "#374151", "icon": "🗑️", "label": "Residu"},
}
 
 
def _h(s: str) -> str:
    """Collapse whitespace so Markdown never treats indented HTML as a code block."""
    return " ".join(s.split())
 
 
CSS = _h("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');
html, body, .stApp, .stMarkdown, button, input, textarea { font-family: 'Poppins','Segoe UI',sans-serif; }
.stApp { background: #F3F6FA; }
.block-container { max-width: 520px; padding-top: 1.2rem; padding-bottom: 3rem; }
.card { background:#fff; border-radius:18px; padding:16px 18px; margin:0 0 14px 0;
        box-shadow:0 2px 12px rgba(16,24,40,.08); }
.sec { font-weight:700; font-size:1rem; color:#111827; margin:8px 0 8px; }
.muted { color:#6B7280; font-size:.82rem; line-height:1.45; }
.hero-blue { background:linear-gradient(135deg,#2F5BEA 0%,#3E7BFA 100%); border-radius:22px;
             padding:20px 20px 22px; margin-bottom:14px; position:relative; overflow:hidden; color:#fff; }
.hero-blue .eyebrow { font-size:.7rem; font-weight:600; letter-spacing:.09em; opacity:.92; }
.hero-blue .title { font-size:1.5rem; font-weight:800; line-height:1.3; margin:6px 0 0; }
.hero-blue .hl { display:inline-block; background:#FBBF24; color:#1F2937; padding:2px 10px;
                 border-radius:6px; margin-top:6px; }
.hero-blue .icon { position:absolute; right:16px; bottom:8px; font-size:3.4rem; }
.hero-teal { background:linear-gradient(135deg,#0FA3A3 0%,#2BC48A 100%); border-radius:20px;
             padding:16px 18px; margin-bottom:14px; color:#fff; }
.hero-teal .sub { font-size:.78rem; opacity:.95; }
.hero-teal .big { font-size:1.25rem; font-weight:700; margin:2px 0 6px; }
.chip { display:inline-block; background:#fff; color:#1F2937; border-radius:999px;
        padding:4px 12px; font-size:.78rem; font-weight:600; margin:4px 6px 0 0; }
.dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; }
div.stButton > button { width:100%; border-radius:16px; padding:.85rem .6rem; font-weight:600;
        border:1.5px solid #E3E8EF; background:#fff; color:#1F2937;
        box-shadow:0 2px 10px rgba(16,24,40,.06); }
div.stButton > button[kind="primary"], div.stButton > button[data-testid="stBaseButton-primary"] {
        background:#16A34A; border-color:#16A34A; color:#fff; }
.pill { display:inline-block; border-radius:999px; padding:6px 16px; font-weight:700; font-size:1rem; }
.conf { font-size:.9rem; font-weight:600; color:#111827; margin-top:8px; }
.err-box { border:1.5px solid #EF4444; background:#FFF5F5; border-radius:12px;
           padding:12px 14px; margin:0 0 14px 0; }
.err-title { color:#B42318; font-weight:700; font-size:.92rem; }
.err-msg { color:#DC2626; font-size:.82rem; margin-top:4px; line-height:1.45; }
.warn-box { border:1.5px solid #F59E0B; background:#FFFBEB; border-radius:12px;
            padding:12px 14px; margin:0 0 14px 0; }
.warn-title { color:#92400E; font-weight:700; font-size:.92rem; }
.warn-msg { color:#B45309; font-size:.82rem; margin-top:4px; line-height:1.45; }
.bar-row { display:flex; align-items:center; gap:8px; margin:6px 0; font-size:.8rem; }
.bar-lab { width:84px; color:#374151; }
.bar { flex:1; height:9px; background:#EEF2F6; border-radius:99px; overflow:hidden; }
.fill { height:100%; border-radius:99px; }
.bar-val { width:44px; text-align:right; color:#374151; font-weight:600; }
ol.steps { margin:10px 0 0 0; padding-left:20px; font-size:.88rem; color:#1F2937; line-height:1.55; }
ol.steps li { margin-bottom:6px; }
</style>
""")
 
 
def hero_blue(banner_path=None) -> str:
    """Blue banner (yellow highlighted title). Uses assets/banner.png if it exists."""
    if banner_path is not None and Path(banner_path).exists():
        b64 = base64.b64encode(Path(banner_path).read_bytes()).decode()
        return _h('<div class="card" style="padding:0;overflow:hidden">'
                  f'<img alt="banner" src="data:image/png;base64,{b64}" '
                  'style="width:100%;display:block"/></div>')
    return _h(
        '<div class="hero-blue"><div class="eyebrow">PROYEK BERBASIS OPENCV</div>'
        '<div class="title">Asisten Klasifikasi<br/>'
        '<span class="hl">dan Pemilahan Sampah</span></div>'
        '<div class="icon">♻️</div></div>')
 
 
def hero_chips() -> str:
    chips = "".join(
        f'<span class="chip"><span class="dot" style="background:{s["color"]}"></span>{s["label"]}</span>'
        for s in CAT_STYLE.values())
    return _h('<div class="hero-teal"><div class="sub">Pilah dari sumber</div>'
              f'<div class="big">4 kategori sampah</div>{chips}</div>')
 
 
def section(title: str) -> str:
    return f'<div class="sec">{html.escape(title)}</div>'
 
 
def result_card(cat: str, item: str, conf: float) -> str:
    s = CAT_STYLE[cat]
    return _h(
        f'<div class="card" style="border-left:6px solid {s["color"]}">'
        f'<span class="pill" style="background:{s["bg"]};color:{s["text"]}">{s["icon"]} {s["label"]}</span>'
        f'<div class="muted" style="margin-top:10px">Perkiraan jenis barang: <b>{html.escape(item)}</b></div>'
        f'<div class="conf">Keyakinan model: {conf:.0%}</div>'
        '<div class="muted">Keyakinan adalah skor model, bukan peluang jawabannya benar.</div></div>')
 
 
def err_box(title: str, msg: str) -> str:
    return _h(f'<div class="err-box"><div class="err-title">{html.escape(title)}</div>'
              f'<div class="err-msg">{html.escape(msg)}</div></div>')
 
 
def warn_box(title: str, msg: str) -> str:
    return _h(f'<div class="warn-box"><div class="warn-title">{html.escape(title)}</div>'
              f'<div class="warn-msg">{html.escape(msg)}</div></div>')
 
 
def bars(category_probs: dict) -> str:
    rows = []
    for cat, p in category_probs.items():
        s = CAT_STYLE[cat]
        rows.append(
            f'<div class="bar-row"><span class="bar-lab">{s["label"]}</span>'
            f'<div class="bar"><div class="fill" style="width:{p * 100:.1f}%;background:{s["color"]}"></div></div>'
            f'<span class="bar-val">{p:.0%}</span></div>')
    return _h('<div class="card">' + "".join(rows) + '</div>')
 
 
def guidance_card(cat: str, title: str, summary: str, steps, manual: bool = False) -> str:
    s = CAT_STYLE[cat]
    note = ('<div class="muted" style="margin-bottom:6px">Panduan berdasarkan pilihan Anda, '
            'bukan prediksi model.</div>') if manual else ""
    lis = "".join(f"<li>{html.escape(t)}</li>" for t in steps)
    return _h(f'<div class="card" style="border-top:5px solid {s["color"]}">{note}'
              f'<div class="sec" style="margin-top:0">{html.escape(title)}</div>'
              f'<div class="muted">{html.escape(summary)}</div><ol class="steps">{lis}</ol></div>')