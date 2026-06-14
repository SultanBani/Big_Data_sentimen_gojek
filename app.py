import streamlit as st
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURASI HALAMAN — wajib dipanggil PERTAMA
# ============================================================
st.set_page_config(
    page_title="Gojek Sentiment & Topic Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS — Nuansa Gradient Modern (Cyan to Blue)
# ============================================================
st.markdown("""
<style>
/* ---- Google Font ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ---- Global ---- */
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(-45deg, #F3F9A7, #A0E8AF, #7AD9EC, #4CB8C4);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: #1f2937;
}
[data-testid="stSidebar"] hr { border-color: rgba(0,0,0,0.1); }

/* ---- Header strip ---- */
.header-strip {
    background: linear-gradient(-45deg, #F3F9A7, #A0E8AF, #7AD9EC, #4CB8C4);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: #1f2937;
    padding: 28px 36px;
    border-radius: 16px;
    margin-bottom: 28px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    border: 1px solid rgba(255, 255, 255, 0.5);
}
.header-strip h1 { margin: 0; font-size: 26px; font-weight: 800; letter-spacing: -0.5px; }
.header-strip p  { margin: 4px 0 0; font-size: 14px; opacity: .85; }

/* ---- Metric cards ---- */
.metric-row { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 160px;
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(10px);
    border-radius: 14px;
    padding: 18px 22px;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.8);
    border-top: 4px solid #4CB8C4;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px rgba(31, 38, 135, 0.1);
}
.metric-card .label  { font-size: 12px; color: #6b7280; font-weight: 600; text-transform: uppercase; letter-spacing: .5px; }
.metric-card .value  { font-size: 26px; font-weight: 800; color: #0284C7; margin-top: 4px; }
.metric-card .sub    { font-size: 11px; color: #9ca3af; margin-top: 2px; }

/* ---- Result banners ---- */
.result-positive {
    background: linear-gradient(135deg, #d1fae5, #a7f3d0);
    border-left: 6px solid #059669;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 16px 0;
}
.result-neutral {
    background: linear-gradient(135deg, #fef9c3, #fde68a);
    border-left: 6px solid #d97706;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 16px 0;
}
.result-negative {
    background: linear-gradient(135deg, #fee2e2, #fca5a5);
    border-left: 6px solid #dc2626;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 16px 0;
}
.result-title { font-size: 20px; font-weight: 800; margin-bottom: 4px; }
.result-desc  { font-size: 14px; opacity: .85; color: #374151; }

/* ---- Topic card ---- */
.topic-card {
    background: linear-gradient(135deg, #eff6ff, #dbeafe);
    border: 1px solid #93c5fd;
    border-radius: 14px;
    padding: 20px 26px;
    margin-top: 18px;
    box-shadow: 0 2px 10px rgba(59,130,246,0.12);
}
.topic-title { font-size: 16px; font-weight: 700; color: #1d4ed8; margin-bottom: 8px; }
.topic-badge {
    display: inline-block;
    background: #1d4ed8;
    color: white;
    padding: 4px 14px;
    border-radius: 99px;
    font-size: 12px;
    font-weight: 700;
    margin-right: 8px;
}
.progress-bar-wrap {
    background: #e0e7ff;
    border-radius: 99px;
    height: 12px;
    margin-top: 10px;
    overflow: hidden;
}
.progress-bar-fill {
    height: 12px;
    border-radius: 99px;
    background: linear-gradient(-45deg, #F3F9A7, #A0E8AF, #7AD9EC, #4CB8C4);
    background-size: 300% 300%;
    animation: gradientBG 10s ease infinite;
    transition: width 0.6s ease;
}

/* ---- Section title ---- */
.section-title {
    font-size: 18px;
    font-weight: 700;
    color: #0284C7;
    margin: 30px 0 14px;
    padding-bottom: 8px;
    border-bottom: 2px solid #bae6fd;
}

/* ---- Chart card ---- */
.chart-card {
    background: white;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 2px 14px rgba(0,0,0,0.06);
    margin-bottom: 24px;
}

/* ---- Error card ---- */
.error-card {
    background: #fff1f2;
    border: 1px solid #fda4af;
    border-radius: 14px;
    padding: 24px;
    text-align: center;
}

/* ---- Member list ---- */
.member-item {
    background: rgba(255,255,255,0.15);
    border-radius: 10px;
    padding: 8px 12px;
    margin-bottom: 6px;
    font-size: 13px;
    font-weight: 500;
}

/* ---- Hide Streamlit branding ---- */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# FUNGSI MEMUAT MODEL — cache agar tidak reload tiap interaksi
# ============================================================
@st.cache_resource(show_spinner=False)
def load_models():
    """Load semua file .pkl model. Raise FileNotFoundError jika tidak ditemukan."""
    with open('svm_final_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vec = pickle.load(f)
    with open('lda_model.pkl', 'rb') as f:
        lda_model = pickle.load(f)
    with open('tf_vectorizer.pkl', 'rb') as f:
        tf_vec = pickle.load(f)
    return svm_model, tfidf_vec, lda_model, tf_vec


# ---- Load model dengan penanganan error ----
models_loaded = False
try:
    with st.spinner("Memuat model analitik... harap tunggu sebentar."):
        svm_model, tfidf_vec, lda_model, tf_vec = load_models()
    models_loaded = True
except FileNotFoundError as e:
    st.markdown("""
    <div class="error-card">
        <h2>File Model Tidak Ditemukan</h2>
        <p>Pastikan keempat file berikut berada di direktori yang sama dengan <code>app.py</code>:</p>
        <ul style="text-align:left; display:inline-block;">
            <li><code>svm_final_model.pkl</code></li>
            <li><code>tfidf_vectorizer.pkl</code></li>
            <li><code>lda_model.pkl</code></li>
            <li><code>tf_vectorizer.pkl</code></li>
        </ul>
        <p style="color:#ef4444; font-size:13px;">Detail error: {}</p>
    </div>
    """.format(str(e)), unsafe_allow_html=True)
    st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")
    st.stop()


# ============================================================
# SIDEBAR NAVIGASI
# ============================================================
with st.sidebar:
    # ---- Judul Proyek ----
    st.markdown("""
    <div style="font-size:14px; font-weight:800; line-height:1.5; margin-bottom:4px; margin-top:10px;">
        Analisis Sentimen &amp; Ekstraksi Topik
    </div>
    <div style="font-size:12px; opacity:.9; margin-bottom:16px; line-height:1.4;">
        SVM + LDA · Ulasan Pengguna Gojek
    </div>
    """, unsafe_allow_html=True)

    # ---- Navigasi Halaman ----
    menu = st.radio(
        "Menu Navigasi",
        ["Testing Prediksi Ulasan", "Dashboard Analisis Topik"]
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---- Anggota Kelompok ----
    st.markdown("<div style='font-size:12px; font-weight:700; margin-bottom:8px; opacity:.9;'>Anggota Kelompok</div>", unsafe_allow_html=True)

    members = [
        ("Arfan Andhika P", "2315061019"),
        ("Sultan Bani Hakim", "2315061103"),
        ("Muhammad Farhan", "2315061083"),
        ("M. Aqsha Fadilah J", "2315061127"),
    ]
    for name, nim in members:
        st.markdown(f"""
        <div class="member-item">
            <div style="font-weight:600; font-size:12px;">{name}</div>
            <div style="font-size:10px; opacity:.75;">NIM: {nim}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---- Instansi ----
    st.markdown("""
    <div style="font-size:11px; text-align:center; opacity:.8; line-height:1.6;">
        <strong>Prodi Teknik Informatika</strong><br>
        Universitas Lampung<br>
        <span style="opacity:.65;">Big Data · 2025/2026</span>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# ────────────────────────────────────────────────────────────
# HALAMAN 1 — Testing Prediksi Ulasan
# ────────────────────────────────────────────────────────────
# ============================================================
if menu == "Testing Prediksi Ulasan":

    # Header
    st.markdown("""
    <div class="header-strip">
        <h1>Testing Prediksi Sentimen Ulasan Real-Time</h1>
        <p>Uji model SVM (Balanced Accuracy 88%) · Ekstraksi topik otomatis via LDA untuk ulasan negatif</p>
    </div>
    """, unsafe_allow_html=True)

    # Metric cards
    st.markdown("""
    <div class="metric-row">
        <div class="metric-card">
            <div class="label">Model Klasifikasi</div>
            <div class="value">LinearSVC</div>
            <div class="sub">class_weight = 'balanced'</div>
        </div>
        <div class="metric-card">
            <div class="label">Balanced Accuracy</div>
            <div class="value">88.00%</div>
            <div class="sub">Evaluasi pada data uji</div>
        </div>
        <div class="metric-card">
            <div class="label">Fitur TF-IDF</div>
            <div class="value">5.000</div>
            <div class="sub">max_features TfidfVectorizer</div>
        </div>
        <div class="metric-card">
            <div class="label">Topik LDA</div>
            <div class="value">3 Topik</div>
            <div class="sub">Khusus ulasan negatif</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    def set_contoh():
        st.session_state["teks_ulasan"] = (
            "Aplikasi gojek makin parah, sering error dan force close. "
            "Driver susah banget dapet, udah nunggu setengah jam ga ada yang mau ambil. "
            "Tarif juga naik terus tapi pelayanan makin buruk. Kecewa banget!"
        )

    # ---- Input Teks ----
    st.markdown('<div class="section-title">Masukkan Teks Ulasan</div>', unsafe_allow_html=True)

    col_input, col_info = st.columns([3, 1])

    with col_input:
        teks_input = st.text_area(
            label="Ketik atau tempel ulasan di bawah ini:",
            placeholder="Contoh: Driver ramah dan tepat waktu, tapi aplikasinya sering force close dan loading lama banget...",
            height=160,
            key="teks_ulasan",
            label_visibility="collapsed",
        )

    with col_info:
        st.markdown("""
        <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:12px; padding:16px; height:160px; display:flex; flex-direction:column; justify-content:center;">
            <div style="font-weight:700; color:#166534; font-size:13px; margin-bottom:8px;">Panduan Input</div>
            <div style="font-size:12px; color:#374151; line-height:1.6;">
                • Masukkan 1 ulasan penuh<br>
                • Bahasa Indonesia atau campuran<br>
                • Minimal beberapa kata<br>
                • Otomatis mendeteksi topik jika <b>Negatif</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---- Tombol Analisis ----
    col_btn, col_clear = st.columns([2, 1])
    with col_btn:
        btn_analisis = st.button(
            "Analisis Sentimen Sekarang",
            type="primary",
            use_container_width=True,
            key="btn_analisis",
        )
    with col_clear:
        btn_contoh = st.button(
            "Isi Contoh Negatif",
            on_click=set_contoh,
            use_container_width=True,
            key="btn_contoh",
        )

    # ---- Proses Prediksi ----
    if btn_analisis:
        teks = teks_input.strip() if teks_input else ""

        if not teks:
            st.warning("Silakan masukkan teks ulasan terlebih dahulu sebelum menganalisis.")
        else:
            st.markdown('<div class="section-title">Hasil Analisis</div>', unsafe_allow_html=True)

            with st.spinner("Model sedang menganalisis teks..."):
                # Vektorisasi TF-IDF
                teks_tfidf = tfidf_vec.transform([teks])
                # Prediksi sentimen
                prediksi = svm_model.predict(teks_tfidf)[0]

            # ── Tampilkan hasil berdasarkan sentimen ──────────────────
            if prediksi == "Positif":
                st.markdown("""
                <div class="result-positive">
                    <div class="result-title">SENTIMEN: POSITIF</div>
                    <div class="result-desc">
                        Model mendeteksi ulasan ini berisi <strong>apresiasi atau kepuasan</strong> terhadap layanan Gojek.
                        Tidak ada ekstraksi topik LDA karena hanya dilakukan untuk ulasan negatif.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            elif prediksi == "Netral":
                st.markdown("""
                <div class="result-neutral">
                    <div class="result-title">SENTIMEN: NETRAL</div>
                    <div class="result-desc">
                        Model mendeteksi ulasan ini bersifat <strong>netral</strong> — mungkin berisi pertanyaan,
                        informasi umum, atau ekspresi yang tidak terlalu positif maupun negatif.
                        Tidak ada ekstraksi topik LDA karena hanya dilakukan untuk ulasan negatif.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:  # Negatif
                st.markdown("""
                <div class="result-negative">
                    <div class="result-title">SENTIMEN: NEGATIF (Keluhan)</div>
                    <div class="result-desc">
                        Model mendeteksi ulasan ini mengandung <strong>keluhan atau ketidakpuasan</strong>.
                        Sistem akan otomatis mengekstraksi topik keluhan menggunakan model LDA di bawah ini.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Ekstraksi Topik LDA ────────────────────────────────
                st.markdown('<div class="section-title">Ekstraksi Topik Keluhan (LDA)</div>', unsafe_allow_html=True)

                with st.spinner("Menganalisis distribusi topik..."):
                    teks_tf    = tf_vec.transform([teks])
                    dist_topik = lda_model.transform(teks_tf)[0]

                topik_labels = {
                    0: ("Topik 1", "Masalah Teknis Aplikasi",
                        "Keluhan terkait bug, force close, error, loading lambat, atau gangguan fitur aplikasi."),
                    1: ("Topik 2", "Layanan & Kualitas Driver",
                        "Keluhan terkait driver tidak mau ambil orderan, sikap driver, atau ketidaksesuaian layanan."),
                    2: ("Topik 3", "Tarif & Kebijakan Harga",
                        "Keluhan terkait harga yang terlalu mahal, kenaikan tarif, atau ketidakadilan biaya layanan."),
                }

                topik_terkuat_idx = int(np.argmax(dist_topik))
                label, nama_topik, deskripsi = topik_labels[topik_terkuat_idx]
                persen_terkuat = dist_topik[topik_terkuat_idx] * 100

                st.markdown(f"""
                <div class="topic-card">
                    <div class="topic-title">Klasifikasi Topik Keluhan</div>
                    <span class="topic-badge">{label}</span>
                    <strong style="font-size:15px; color:#1e3a8a;">{nama_topik}</strong>
                    <p style="font-size:13px; color:#4b5563; margin-top:10px;">{deskripsi}</p>
                    <div style="font-size:12px; color:#6b7280; margin-top:6px;">
                        Tingkat kedekatan ke topik ini:
                        <strong style="color:#1d4ed8;">{persen_terkuat:.1f}%</strong>
                    </div>
                    <div class="progress-bar-wrap">
                        <div class="progress-bar-fill" style="width:{min(persen_terkuat, 100):.1f}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Detail distribusi semua topik ────────────────────
                st.markdown('<div class="section-title">Distribusi Probabilitas ke Semua Topik</div>', unsafe_allow_html=True)

                cols = st.columns(3)
                warna_topik = ["#dc2626", "#d97706", "#7c3aed"]
                warna_bg    = ["#fff1f2", "#fffbeb", "#f5f3ff"]
                warna_border= ["#fca5a5", "#fde68a", "#c4b5fd"]

                for i, col in enumerate(cols):
                    lbl, nm, _ = topik_labels[i]
                    pct = dist_topik[i] * 100
                    is_top = (i == topik_terkuat_idx)
                    border_style = "3px solid " + warna_topik[i] if is_top else "1px solid " + warna_border[i]
                    with col:
                        st.markdown(f"""
                        <div style="background:{warna_bg[i]}; border:{border_style}; border-radius:12px; padding:16px; text-align:center;">
                            <div style="font-size:11px; font-weight:700; color:{warna_topik[i]}; text-transform:uppercase; letter-spacing:.5px;">{lbl}</div>
                            <div style="font-size:22px; font-weight:800; color:{warna_topik[i]}; margin:8px 0;">{pct:.1f}%</div>
                            <div style="font-size:11px; color:#4b5563; font-weight:500;">{nm}</div>
                            {"<div style='margin-top:6px; font-size:10px; font-weight:700; color:" + warna_topik[i] + ";'>TOPIK DOMINAN</div>" if is_top else ""}
                        </div>
                        """, unsafe_allow_html=True)

    # ---- Catatan metodologi ----
    with st.expander("Tentang Metodologi & Pipeline Model"):
        st.markdown("""
        **Pipeline Preprocessing Teks:**
        1. Data ulasan dibersihkan dari karakter non-alfanumerik, URL, dan angka
        2. Konversi ke huruf kecil (*lowercase*)
        3. Penghapusan *stopword* Bahasa Indonesia
        4. Normalisasi kata-kata tidak baku (slang/typo)

        **Model SVM (Support Vector Machine):**
        - Algoritma: `LinearSVC` dari scikit-learn
        - Vectorizer: `TfidfVectorizer` dengan `max_features=5000`
        - Parameter kunci: `class_weight='balanced'` untuk mengatasi ketidakseimbangan kelas
        - Target: **Balanced Accuracy ≈ 88%** (lebih fokus pada recall kelas Negatif)

        **Model LDA (Latent Dirichlet Allocation):**
        - Hanya dijalankan pada prediksi **Negatif**
        - Vectorizer: `CountVectorizer` (bag-of-words)
        - Parameter: `n_components=3` (3 topik utama)
        - Output: Distribusi probabilitas teks ke setiap topik
        """)


# ============================================================
# ────────────────────────────────────────────────────────────
# HALAMAN 2 — Dashboard Analisis Topik (SEMUA CHART SEKALIGUS)
# ────────────────────────────────────────────────────────────
# ============================================================
else:
    # Header
    st.markdown("""
    <div class="header-strip">
        <h1>Dashboard Analisis Topik Keluhan</h1>
        <p>Pemetaan ilmiah akar masalah utama berdasarkan ulasan negatif pengguna Gojek · Model LDA · 3 Topik Utama</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Metric Cards ----
    st.markdown("""
    <div class="metric-row">
        <div class="metric-card">
            <div class="label">Total Klaster Topik</div>
            <div class="value">3 Topik</div>
            <div class="sub">Hasil LDA dari ulasan negatif</div>
        </div>
        <div class="metric-card">
            <div class="label">Balanced Accuracy SVM</div>
            <div class="value">88.00%</div>
            <div class="sub">class_weight = 'balanced'</div>
        </div>
        <div class="metric-card">
            <div class="label">Dataset Ulasan</div>
            <div class="value">100.000</div>
            <div class="sub">Google Play Store Gojek</div>
        </div>
        <div class="metric-card">
            <div class="label">Fokus Analisis LDA</div>
            <div class="value" style="color:#dc2626;">Negatif</div>
            <div class="sub">Hanya ulasan bersentimen negatif</div>
        </div>
        <div class="metric-card">
            <div class="label">Fitur TF-IDF</div>
            <div class="value">5.000</div>
            <div class="sub">TfidfVectorizer max_features</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Ambil data dari model ----
    fitur_kata   = tf_vec.get_feature_names_out()
    topik_names  = ["Masalah Teknis Aplikasi", "Layanan & Kualitas Driver", "Tarif & Kebijakan Harga"]
    warna_charts = ["Reds_r", "Oranges_r", "Purples_r"]
    warna_wc     = ["Reds", "Oranges", "Purples"]
    warna_hex    = ["#dc2626", "#d97706", "#7c3aed"]

    # ============================================================
    # BAGIAN 1 — Bar Chart: 10 Kata Kunci Terpenting per Topik
    # ============================================================
    st.markdown('<div class="section-title">Bagian 1 — Bar Chart: 10 Kata Kunci Terpenting per Topik</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px; padding:14px 18px; margin-bottom:20px; font-size:13px; color:#166534;">
        <strong>Cara membaca:</strong> Sumbu X menunjukkan bobot kata dalam topik LDA.
        Semakin besar bobot, semakin <em>karakteristik</em> kata tersebut mendefinisikan topik tersebut.
    </div>
    """, unsafe_allow_html=True)

    # Plot tiga bar chart dalam satu row
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)

    fig_bar, axes_bar = plt.subplots(1, 3, figsize=(18, 6))
    fig_bar.patch.set_facecolor('#ffffff')

    for idx, topic in enumerate(lda_model.components_):
        top_idx   = topic.argsort()[:-11:-1]
        top_kata  = [fitur_kata[i] for i in top_idx]
        top_bobot = topic[top_idx]

        ax = axes_bar[idx]
        ax.set_facecolor('#fafafa')

        bars = ax.barh(
            top_kata[::-1],
            top_bobot[::-1],
            color=matplotlib.colormaps[warna_charts[idx]](
                np.linspace(0.35, 0.75, 10)
            ),
            edgecolor='white',
            linewidth=0.8,
            height=0.7,
        )

        # Tambahkan label nilai di ujung bar
        for bar_obj, val in zip(bars, top_bobot[::-1]):
            ax.text(
                bar_obj.get_width() + max(top_bobot) * 0.01,
                bar_obj.get_y() + bar_obj.get_height() / 2,
                f"{val:.0f}",
                va='center', ha='left',
                fontsize=8, color='#374151', fontweight='500'
            )

        ax.set_title(
            f"Topik {idx+1}\\n{topik_names[idx]}",
            fontsize=13, fontweight='bold', color=warna_hex[idx], pad=12
        )
        ax.set_xlabel("Bobot Kata (Skor LDA)", fontsize=10, color='#6b7280')
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='x', labelsize=8, colors='#9ca3af')
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color('#e5e7eb')
        ax.grid(axis='x', linestyle='--', alpha=0.4, color='#d1d5db')
        ax.set_xlim(0, max(top_bobot) * 1.18)

    plt.suptitle(
        "10 Kata Kunci Terpenting per Topik — Model LDA",
        fontsize=15, fontweight='bold', color='#111827', y=1.02
    )
    plt.tight_layout(pad=2.5)
    st.pyplot(fig_bar, use_container_width=False)
    plt.close(fig_bar)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---- Tabel ringkasan kata kunci ----
    with st.expander("Lihat Tabel Kata Kunci per Topik (Top 10)"):
        import pandas as pd
        col_t1, col_t2, col_t3 = st.columns(3)
        for idx, (col, topic) in enumerate(zip([col_t1, col_t2, col_t3], lda_model.components_)):
            top_idx  = topic.argsort()[:-11:-1]
            top_kata = [fitur_kata[i] for i in top_idx]
            top_bdt  = [round(topic[i], 1) for i in top_idx]
            df_tmp   = pd.DataFrame({"Kata": top_kata, "Bobot": top_bdt})
            with col:
                st.markdown(f"**Topik {idx+1} — {topik_names[idx]}**")
                st.dataframe(df_tmp, hide_index=True, use_container_width=True)

    # ============================================================
    # BAGIAN 2 — WordCloud: Awan Kata per Topik
    # ============================================================
    st.markdown('<div class="section-title">Bagian 2 — WordCloud: Awan Kata per Topik</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#eff6ff; border:1px solid #93c5fd; border-radius:10px; padding:14px 18px; margin-bottom:20px; font-size:13px; color:#1e40af;">
        <strong>Cara membaca:</strong> Semakin besar ukuran kata dalam WordCloud,
        semakin tinggi bobotnya dalam mendefinisikan topik tersebut.
        Warna digunakan untuk estetika visual, bukan representasi makna.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)

    fig_wc, axes_wc = plt.subplots(1, 3, figsize=(18, 6))
    fig_wc.patch.set_facecolor('#ffffff')

    for idx, topic in enumerate(lda_model.components_):
        # Ambil 50 kata teratas untuk WordCloud
        top_idx_wc  = topic.argsort()[:-51:-1]
        dict_kata   = {fitur_kata[i]: float(topic[i]) for i in top_idx_wc}

        wc = WordCloud(
            width=600,
            height=400,
            background_color='white',
            colormap=warna_wc[idx],
            max_words=50,
            min_font_size=10,
            max_font_size=90,
            prefer_horizontal=0.85,
            margin=8,
        )
        wc.generate_from_frequencies(dict_kata)

        ax = axes_wc[idx]
        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(
            f"Topik {idx+1} — {topik_names[idx]}",
            fontsize=13, fontweight='bold', color=warna_hex[idx], pad=12
        )
        ax.axis('off')

        # Border bawah berwarna
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.suptitle(
        "WordCloud: Awan Kata 3 Topik Keluhan Pengguna Gojek",
        fontsize=15, fontweight='bold', color='#111827', y=1.02
    )
    plt.tight_layout(pad=2.5)
    st.pyplot(fig_wc, use_container_width=False)
    plt.close(fig_wc)

    st.markdown('</div>', unsafe_allow_html=True)

    # ============================================================
    # BAGIAN 3 — Ringkasan Deskriptif Topik
    # ============================================================
    st.markdown('<div class="section-title">Bagian 3 — Ringkasan Deskriptif Setiap Topik</div>', unsafe_allow_html=True)

    deskripsi_topik = [
        {
            "warna": "#dc2626",
            "bg": "#fff1f2",
            "border": "#fca5a5",
            "judul": "Topik 1 — Masalah Teknis Aplikasi",
            "isi": (
                "Topik ini mendominasi keluhan yang berkaitan dengan performa dan stabilitas aplikasi Gojek. "
                "Pengguna sering mengeluhkan aplikasi yang <strong>sering force close</strong>, "
                "<strong>loading lama</strong>, <strong>error</strong>, dan <strong>bug</strong> pada berbagai fitur. "
                "Keluhan ini mengindikasikan perlunya peningkatan kualitas kode dan infrastruktur server."
            ),
            "kata_kunci": ["aplikasi", "error", "force close", "loading", "bug", "update", "crash"],
        },
        {
            "warna": "#d97706",
            "bg": "#fffbeb",
            "border": "#fde68a",
            "judul": "Topik 2 — Layanan & Kualitas Driver",
            "isi": (
                "Topik ini mengelompokkan keluhan terkait pengalaman dengan <strong>driver</strong> dan <strong>kualitas layanan</strong>. "
                "Pengguna mengeluhkan driver yang <strong>sulit ditemukan</strong>, <strong>tidak mau mengambil orderan</strong>, "
                "waktu tunggu yang terlalu lama, atau sikap driver yang tidak memuaskan. "
                "Ini menjadi perhatian utama pada aspek <em>human capital</em> Gojek."
            ),
            "kata_kunci": ["driver", "orderan", "tunggu", "cancel", "tidak", "lama", "susah"],
        },
        {
            "warna": "#7c3aed",
            "bg": "#f5f3ff",
            "border": "#c4b5fd",
            "judul": "Topik 3 — Tarif & Kebijakan Harga",
            "isi": (
                "Topik ini berisi keluhan seputar <strong>kenaikan harga</strong> dan <strong>ketidakadilan tarif</strong>. "
                "Pengguna merasa tarif Gojek semakin mahal, tidak sebanding dengan kualitas layanan yang diterima, "
                "atau ada <strong>biaya tersembunyi</strong> yang tidak transparan. "
                "Ini mencerminkan sensitivitas pengguna terhadap perubahan kebijakan harga."
            ),
            "kata_kunci": ["harga", "mahal", "tarif", "biaya", "naik", "murah", "promo"],
        },
    ]

    for d in deskripsi_topik:
        st.markdown(f"""
        <div style="background:{d['bg']}; border:1px solid {d['border']}; border-left:5px solid {d['warna']};
                    border-radius:14px; padding:20px 24px; margin-bottom:16px;">
            <div style="font-size:16px; font-weight:800; color:{d['warna']}; margin-bottom:8px;">
                {d['judul']}
            </div>
            <p style="font-size:13.5px; color:#374151; line-height:1.7; margin:0 0 12px;">
                {d['isi']}
            </p>
            <div style="font-size:12px; font-weight:600; color:{d['warna']}; margin-bottom:6px;">
                Kata kunci representatif:
            </div>
            <div>
                {''.join([f'<span style="background:{d["warna"]}22; color:{d["warna"]}; border:1px solid {d["warna"]}44; border-radius:99px; padding:3px 10px; font-size:11px; font-weight:600; margin-right:6px; display:inline-block; margin-bottom:4px;">{k}</span>' for k in d['kata_kunci']])}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---- Footer ----
    st.markdown("""
    <div style="text-align:center; margin-top:40px; padding:20px; border-top:1px solid #e5e7eb; color:#9ca3af; font-size:12px;">
        Dibangun menggunakan <strong>Streamlit</strong> · 
        <strong>scikit-learn</strong> · <strong>WordCloud</strong> · <strong>Seaborn/Matplotlib</strong><br>
        Proyek Big Data — Program Studi Teknik Informatika, Universitas Lampung · 2025/2026
    </div>
    """, unsafe_allow_html=True)