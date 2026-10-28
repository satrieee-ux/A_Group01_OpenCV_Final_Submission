"""Phase 2, Step 7: handling guidance per category (text only, no model logic).

Source tags:
  INGUB = Instruksi Gubernur DKI Jakarta No. 5 Tahun 2026, Lampiran A
          (primary source, but valid for DKI Jakarta only)
  SEK-n = secondary source (article / journal / campus SOP): verify before the report
  UMUM  = general advice WITHOUT a source: add a source or remove it
Run  python -m src.guidance_v2  to print everything.
"""

SOURCES = {
    "INGUB": "Instruksi Gubernur DKI Jakarta No. 5 Tahun 2026, Lampiran A. "
             "https://jdih.jakarta.go.id/link/download-fulltext/14954",
    "SEK-1": "Halodoc, 'Ini Jenis Jenis Sampah dan Contohnya'. "
             "https://www.halodoc.com/artikel/ini-jenis-jenis-sampah-dan-contohnya",
    "SEK-2": "Halodoc, 'Cara Mudah Pemilahan Sampah di Rumah Sendiri'. "
             "https://www.halodoc.com/artikel/cara-mudah-pemilahan-sampah-agar-rumah-makin-bersih",
    "SEK-3": "CNA Indonesia, 'Jakarta sampah pilah Bantargebang rumah'. "
             "https://www.cna.id/indonesia/jakarta-sampah-pilah-bantargebang-rumah-47101",
    "SEK-4": "SOP pengelolaan limbah B3 IPB (SOP kampus, bukan aturan rumah tangga). "
             "https://sustainability.ipb.ac.id/wp-content/uploads/2025/11/SDG-12_NO.-2-2.pdf",
    "SEK-5": "Paparan sosialisasi Perda Kota Balikpapan No. 4 Tahun 2022 (regional). "
             "https://jdih.balikpapan.go.id/jdih/common/dokumen/"
             "paparansosialisasiperda4tahun2022tentangpengelolaansrtdanssrtdlh.pdf",
    "SEK-6": "Iswanto dkk., Timbulan Sampah B3 Rumah Tangga dan Potensi Dampak Kesehatan "
             "(Jurnal Manusia dan Lingkungan, UGM). https://jurnal.ugm.ac.id/JML/article/download/18789/12120",
}

FOOTER = (
    "Panduan umum. Kategori dan jalur pengolahan mengacu pada Instruksi Gubernur DKI Jakarta "
    "No. 5 Tahun 2026 (berlaku di DKI Jakarta). Aturan di daerah Anda bisa berbeda; "
    "tanyakan Dinas Lingkungan Hidup setempat."
)

GUIDANCE = {
    "organik": {
        "title": "Sampah Organik",
        "summary": "Sampah yang mudah terurai, misalnya sisa makanan, kulit buah, dan daun.",
        "steps": [
            ("Taruh di wadah khusus organik, terpisah dari plastik, logam, kaca, dan bahan kimia.", "SEK-1"),
            ("Olah lewat komposting, budidaya maggot BSF, atau biodigester.", "INGUB"),
        ],
    },
    "anorganik": {
        "title": "Sampah Anorganik",
        "summary": "Sampah yang dapat didaur ulang atau dimanfaatkan kembali, misalnya kertas, kardus, "
                   "botol plastik, botol kaca, kemasan plastik, dan logam.",
        "steps": [
            ("Bersihkan dan keringkan barang sebelum dipisahkan, bila memungkinkan.", "SEK-2"),
            ("Serahkan ke bank sampah atau pengumpul daur ulang (offtaker) terdekat.", "INGUB"),
            ("Plastik yang kotor atau bercampur bahan lain bisa tidak diterima dan berakhir sebagai residu.", "SEK-1"),
        ],
    },
    "b3": {
        "title": "Sampah B3 (Bahan Berbahaya dan Beracun)",
        "summary": "Sampah yang mengandung bahan berbahaya dan beracun, misalnya baterai, bohlam, dan e-waste.",
        "steps": [
            ("Jangan dicampur dengan sampah lain.", "SEK-3"),
            ("Jangan dibakar atau dibuang ke sungai.", "SEK-1"),
            ("Kemas rapi dan terpisah. Baterai dan lampu sebaiknya di wadah terpisah agar lampu tidak pecah.", "SEK-4"),
            ("Serahkan ke TPS B3 atau titik pengumpulan (drop box) B3.", "INGUB, SEK-5"),
            ("Jika tidak tahu lokasinya, tanyakan ke Dinas Lingkungan Hidup setempat.", "UMUM"),
        ],
    },
    "residu": {
        "title": "Sampah Residu",
        "summary": "Sampah yang tidak bisa diolah lewat pengolahan organik maupun daur ulang.",
        "steps": [
            ("Taruh di wadah residu, terpisah dari tiga kategori lain.", "INGUB"),
            ("Residu berujung ke pemrosesan akhir (di DKI Jakarta: RDF atau PLTSa). "
             "Memilah tiga kategori lain terlebih dulu mengurangi volume residu.", "INGUB, SEK-2"),
            ("Pakaian atau kain yang masih layak pakai sebaiknya didonasikan.", "UMUM"),
        ],
    },
}

B3_NOT_RECOGNIZED_INTRO = (
    "Model tidak mengenali barang berikut. Jika barang Anda termasuk jenis ini, "
    "perlakukan sebagai B3 walaupun hasil prediksi menunjukkan kategori lain:"
)
B3_NOT_RECOGNIZED = [
    ("bohlam / lampu", "INGUB"),
    ("kemasan pengharum ruangan, pemutih, pembersih lantai, dan pembasmi serangga", "INGUB"),
    ("kemasan cat dan pestisida", "SEK-6"),
]


def guidance_for(category: str) -> dict:
    if category not in GUIDANCE:
        raise KeyError(f"No guidance for category '{category}'")
    return GUIDANCE[category]


def main():
    for cat, g in GUIDANCE.items():
        print(f"\n[{cat}] {g['title']}\n  {g['summary']}")
        for i, (text, tag) in enumerate(g["steps"], 1):
            print(f"  {i}. {text}  [{tag}]")
    print("\n" + B3_NOT_RECOGNIZED_INTRO)
    for text, tag in B3_NOT_RECOGNIZED:
        print(f"  - {text}  [{tag}]")
    print("\n" + FOOTER)
    untagged = [t for g in GUIDANCE.values() for t, tag in g["steps"] if tag == "UMUM"]
    print(f"\nSteps tagged UMUM (need a source or removal): {len(untagged)}")
    for t in untagged:
        print("  -", t)


if __name__ == "__main__":
    main()