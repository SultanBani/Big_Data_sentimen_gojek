import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Gojek Sentiment & Topic Analytics",
    page_icon="🛵",
    layout="wide"
)

# Custom CSS Nuansa Gojek
st.markdown("""
    <style>
    .main-title { font-size:32px; font-weight:bold; color:#118c4f; text-align:center; margin-bottom:5px; }
    .sub-title { font-size:16px; color:#555555; text-align:center; margin-bottom:30px; }
    .metric-box { padding: 15px; background-color: #f4f6f8; border-radius: 10px; border-left: 5px solid #118c4f; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# FUNGSI MEMUAT MODEL
# ==========================================
@st.cache_resource
def load_models():
    with open('svm_final_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vec = pickle.load(f)
    with open('lda_model.pkl', 'rb') as f:
        lda_model = pickle.load(f)
    with open('tf_vectorizer.pkl', 'rb') as f:
        tf_vec = pickle.load(f)
    return svm_model, tfidf_vec, lda_model, tf_vec

try:
    svm_model, tfidf_vec, lda_model, tf_vec = load_models()
except FileNotFoundError:
    st.error("❌ File model (.pkl) tidak ditemukan. Pastikan keempat file model berada di folder yang sama dengan app.py.")
    st.stop()

# ==========================================
# SIDEBAR / MENU NAVIGASI
# ==========================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/9/9a/Gojek_logo_2019.svg", width=150)
st.sidebar.title("Navigasi Menu")
menu = st.sidebar.radio("Pilih Halaman:", ["Testing Prediksi Ulasan", "Dashboard Analisis Topik"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Anggota Kelompok:**")
st.sidebar.text("1. Arfan Andhika P (2315061019)\n2. Sultan Bani Hakim (2315061103)\n3. Muhammad Farhan (2315061083)\n4. M. Aqsha Fadilah J (2315061127)")
st.sidebar.markdown("**Universitas Lampung**")

# ==========================================
# HALAMAN 1: TESTING PREDIKSI ULASAN
# ==========================================
if menu == "Testing Prediksi Ulasan":
    st.markdown('<div class="main-title">🛵 Web Testing Prediksi Sentimen Ulasan</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Uji kepekaan model SVM dalam mendeteksi sentimen secara real-time</div>', unsafe_allow_html=True)
    
    st.write("### Masukkan Teks Ulasan:")
    teks_input = st.text_area("Ketik ulasan di sini (contoh: 'Driver ramah, tapi aplikasinya lemot dan sering force close...')", height=150)
    
    if st.button("Analisis Sentimen Teks", type="primary"):
        if teks_input.strip() == "":
            st.warning("⚠️ Silakan masukkan teks ulasan terlebih dahulu.")
        else:
            # Prediksi Sentimen (SVM)
            teks_vektor = tfidf_vec.transform([teks_input])
            hasil_prediksi = svm_model.predict(teks_vektor)[0]
            
            st.write("---")
            st.write("### Hasil Klasifikasi:")
            
            if hasil_prediksi == "Positif":
                st.success("🟢 **Sentimen: POSITIF** — Ulasan ini dinilai berisi apresiasi.")
            elif hasil_prediksi == "Netral":
                st.info("🟡 **Sentimen: NETRAL** — Ulasan ini dinilai berisi pertanyaan/informasi biasa.")
            else:
                st.error("🔴 **Sentimen: NEGATIF (Keluhan)** — Model mendeteksi adanya komplain.")
                
                # Ekstraksi Topik (LDA) khusus jika sentimennya Negatif
                teks_tf = tf_vec.transform([teks_input])
                distribusi_topik = lda_model.transform(teks_tf)[0]
                topik_terkuat = np.argmax(distribusi_topik) + 1
                kedekatan = distribusi_topik[topik_terkuat - 1] * 100
                
                st.markdown(f"""
                <div class="metric-box">
                    <strong>🔍 Rekomendasi Fitur Keluhan (LDA):</strong><br>
                    Keluhan di atas terdeteksi masuk ke dalam kelompok <strong>Topik {topik_terkuat}</strong> dengan tingkat keyakinan karakteristik sebesar <strong>{kedekatan:.2f}%</strong>.
                </div>
                """, unsafe_allow_html=True)

# ==========================================
# HALAMAN 2: DASHBOARD ANALISIS TOPIK
# ==========================================
else:
    st.markdown('<div class="main-title">📊 Dashboard Analisis Topik Keluhan</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Pemetaan Ilmiah Akar Masalah Utama Berdasarkan Ulasan Negatif</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-box"><strong>Total Klaster Topik</strong><br><span style="font-size:24px; font-weight:bold; color:#118c4f;">3 Kelompok Utama</span></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-box"><strong>Akurasi SVM Final</strong><br><span style="font-size:24px; font-weight:bold; color:#118c4f;">88.00% (Balanced)</span></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-box"><strong>Fokus Analisis (LDA)</strong><br><span style="font-size:24px; font-weight:bold; color:#d9534f;">Ulasan Negatif Saja</span></div>', unsafe_allow_html=True)
        
    st.write("---")
    pilihan_grafik = st.radio("Pilih Visualisasi Dashboard:", ["Bar Chart (Distribusi Bobot Kata)", "WordCloud (Awan Kata)"], horizontal=True)
    
    fitur_kata = tf_vec.get_feature_names_out()
    
    if pilihan_grafik == "Bar Chart (Distribusi Bobot Kata)":
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        warna_topik = ['Reds_r', 'Blues_r', 'Greens_r']
        
        for idx, topic in enumerate(lda_model.components_):
            top_indeks = topic.argsort()[:-11:-1]
            top_kata = [fitur_kata[i] for i in top_indeks]
            bobot_kata = topic[top_indeks]
            
            ax = axes[idx]
            sns.barplot(x=bobot_kata, y=top_kata, ax=ax, palette=warna_topik[idx])
            ax.set_title(f"Karakteristik Topik {idx+1}", fontsize=14, fontweight='bold')
            ax.set_xlabel("Bobot Kata")
        
        plt.tight_layout()
        st.pyplot(fig)
        
    else:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        tema_warna = ['Reds', 'Blues', 'Greens']
        
        for idx, topic in enumerate(lda_model.components_):
            top_indeks = topic.argsort()[:-31:-1]
            dict_kata = {fitur_kata[i]: topic[i] for i in top_indeks}
            
            wc = WordCloud(width=400, height=300, background_color='white', colormap=tema_warna[idx])
            wc.generate_from_frequencies(dict_kata)
            
            ax = axes[idx]
            ax.imshow(wc, interpolation='bilinear')
            ax.set_title(f"Awan Kata Topik {idx+1}", fontsize=14, fontweight='bold')
            ax.axis('off')
            
        plt.tight_layout()
        st.pyplot(fig)