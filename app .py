import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

st.set_page_config(page_title="Heart Predictor Pro", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #0b1329 50%, #1e1b4b 100%) !important;
        color: #f8fafc !important;
        font-family: 'Segoe UI', system-ui, sans-serif;
    }

    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        border-radius: 14px !important;
        padding: 20px 15px !important;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3), inset 0 0 10px rgba(56, 189, 248, 0.05) !important;
        text-align: center;
    }
    div[data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-size: 34px !important;
        font-weight: 800 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stButton>button {
        background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 30px !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        letter-spacing: 1px !important;
        box-shadow: 0 4px 15px rgba(6, 182, 212, 0.4) !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.6) !important;
    }

    .stSlider, .stSelectbox, .stNumberInput {
        background: rgba(255, 255, 255, 0.02) !important;
        border-radius: 8px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("/content/heart.csv")

try:
    df = load_data()
except Exception as e:
    st.error(f"Gagal memuat dataset lokal '/content/heart.csv'. Pastikan file sudah di-upload ke Colab! Error: {e}")
    st.stop()

X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)
presisi = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

st.sidebar.markdown("<h2 style='text-align:center; color:#38bdf8; font-size:20px; margin-bottom:20px;'>MENU NAVIGASI</h2>", unsafe_allow_html=True)
menu = st.sidebar.radio(
    "",
    ("1. Informasi Proyek", "2. Analisis Data (EDA)", "3. Performa Model", "4. Simulasi Prediksi"),
    index=2
)

plt.style.use('dark_background')

if menu == "1. Informasi Proyek":
    st.markdown("<div style='background:rgba(255,255,255,0.03); padding:30px; border-radius:16px; border:1px solid rgba(255,255,255,0.1); text-align:center;'><h1 style='color:#38bdf8; margin:0;'> HEART PREDICTOR PRO</h1><p style='color:#94a3b8; font-size:16px; margin-top:10px;'>Sistem Cerdas Deteksi Risiko Penyakit Jantung Berbasis Machine Learning</p></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader(" Informasi Pengembangan")
    st.info("Aplikasi Web Resmi Luaran Tugas Akhir / Proyek - Universitas 17 Agustus 1945 Surabaya")

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown("""
**Fitur Utama Aplikasi:**
* Analisis Korelasi Otomatis: Melihat hubungan variabel rekam medis pasien.
* Engine Random Forest: Pemodelan prediktif akurasi tinggi.
* Simulasi Interaktif: Memasukkan parameter klinis secara langsung.
""")
    with col_info2:
        st.markdown("""
**Metodologi Komputasi:**
* Validasi Data: Pencocokan parameter klinis otomatis.
* Split Data: 80% Training Data & 20% Testing Data.
""")

elif menu == "2. Analisis Data (EDA)":
    st.title(" Matriks Korelasi Fitur Medis")
    st.markdown("Peta korelasi antar parameter klinis untuk melihat fitur yang paling memengaruhi target penyakit jantung.")
    st.markdown("---")

    fig1, ax1 = plt.subplots(figsize=(11, 5), facecolor='#0f172a')
    ax1.set_facecolor('#0f172a')
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='Blues', ax=ax1, annot_kws={"size": 9, "weight": "bold"}, cbar=True)
    plt.xticks(rotation=45, ha='right', color='#cbd5e1')
    plt.yticks(color='#cbd5e1')
    plt.tight_layout()
    st.pyplot(fig1)

elif menu == "3. Performa Model":
    st.title(" Evaluasi & Validasi Performa Model")
    st.markdown("---")

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric(label="Akurasi (Accuracy)", value=f"{akurasi*100:.1f}%")
    with m2: st.metric(label="Presisi (Precision)", value=f"{presisi*100:.1f}%")
    with m3: st.metric(label="Recall (Sensitivitas)", value=f"{recall*100:.1f}%")
    with m4: st.metric(label="F1-Score", value=f"{f1*100:.1f}%")

    st.write("")
    st.write("")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.subheader(" Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(5, 3.8), facecolor='#0f172a')
        ax_cm.set_facecolor('#0f172a')

        sns.heatmap(cm, annot=True, fmt='d', cmap='cool', xticklabels=['Aman', 'Risiko'], yticklabels=['Aman', 'Risiko'], ax=ax_cm, cbar=False, annot_kws={"size": 14, "weight":"bold"})

        ax_cm.set_xlabel('Prediksi Model', color='#cbd5e1')
        ax_cm.set_ylabel('Data Aktual', color='#cbd5e1')
        st.pyplot(fig_cm)
    with col_g2:
        st.subheader(" Grafik Perbandingan Metrik")
        fig_bar, ax_bar = plt.subplots(figsize=(5, 3.8), facecolor='#0f172a')
        ax_bar.set_facecolor('#0f172a')
        metrics_names = ['Akurasi', 'Presisi', 'Recall', 'F1']
        metrics_values = [akurasi, presisi, recall, f1]

        bars = ax_bar.bar(metrics_names, metrics_values, color=['#38bdf8', '#0ea5e9', '#0284c7', '#0369a1'], width=0.4, edgecolor='none', linewidth=1)

        ax_bar.set_ylim(0, 1.1)
        for bar in bars:
            yval = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval*100:.1f}%", ha='center', va='bottom', fontweight='bold', color='#38bdf8')
        st.pyplot(fig_bar)

elif menu == "4. Simulasi Prediksi":
    st.markdown("<h2 style='color:#38bdf8; margin:0;'> Ruang Simulasi Diagnostik Pasien</h2>", unsafe_allow_html=True)
    st.markdown("Isi parameter klinis hasil laboratorium pasien di bawah ini secara objektif.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("###  Profil Dasar")
        age = st.slider('Umur Pasien (Age)', 10, 100, 52)
        sex = st.selectbox('Jenis Kelamin', (1, 0), format_func=lambda x: 'Laki-laki' if x==1 else 'Perempuan')
        cp = st.selectbox('Tipe Nyeri Dada (CP)', (0, 1, 2, 3), format_func=lambda x: f"Tipe {x}" if x!=0 else "Typical Angina (0)")
        trestbps = st.number_input('Tekanan Darah Statis (trestbps)', min_value=90, max_value=200, value=130)
    with col2:
        st.markdown("###  Hasil Laboratorium")
        chol = st.number_input('Kadar Kolesterol Serum (chol)', min_value=100, max_value=600, value=240)
        fbs = st.selectbox('Gula Darah Puasa > 120 mg/dl (fbs)', (0, 1), format_func=lambda x: 'Tidak Cenderung (0)' if x==0 else 'Ya, Tinggi (1)')
        restecg = st.selectbox('Elektrokardiografi Istirahat', (0, 1, 2), format_func=lambda x: f"Normal / Tipe {x}" if x!=2 else "Left Ventricular Hypertrophy (2)")
        thalach = st.slider('Detak Jantung Maksimal (thalach)', 60, 220, 150)
    with col3:
        st.markdown("###  Grafik Jantung Eksperimen")
        exang = st.selectbox('Angina Akibat Olahraga (exang)', (0, 1), format_func=lambda x: 'Tidak (0)' if x==0 else 'Ya (1)')
        oldpeak = st.number_input('Depresi ST Segmen (oldpeak)', min_value=0.0, max_value=6.0, value=1.0, step=0.1)
        slope = st.selectbox('Kemiringan Segmen ST (slope)', (0, 1, 2), format_func=lambda x: f"Tipe {x}" if x!=0 else "Upsloping (0)")
        ca = st.selectbox('Jumlah Pembuluh Darah Utama Terwarna (ca)', (0, 1, 2, 3, 4))
        thal = st.selectbox('Status Thalassemia (thal)', (1, 2, 3, 0), format_func=lambda x: f"Tipe {x}" if x!=1 else "Normal (1)")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("PROSES DIAGNOSIS MODEL UNTAG SURABAYA"):
        input_data = pd.DataFrame({
            'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps], 'chol': [chol],
            'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach], 'exang': [exang],
            'oldpeak': [oldpeak], 'slope': [slope], 'ca': [ca], 'thal': [thal]
        })
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        st.markdown("<br><h3 style='color:#ffffff;'> Hasil Keputusan Diagnostik:</h3>", unsafe_allow_html=True)
        if prediction[0] == 1:
            st.error(f" TERINDIKASI RISIKO PENYAKIT JANTUNG (Tingkat Keyakinan: {probability[0][1]*100:.2f}%)")
            st.warning(" REKOMENDASI KLINIS: Hasil komputasi menunjukkan indikasi aktif. Harap segera lakukan evaluasi penunjang medis (seperti Treadmill Test atau Echocardiography) bersama Dokter Spesialis Jantung (Sp.JP).")
        else:
            st.success(f" STATUS: RISIKO RENDAH / AMAN (Tingkat Keyakinan: {probability[0][0]*100:.2f}%)")
            st.info(" REKOMENDASI KESEHATAN: Kondisi jantung diprediksi dalam batas aman. Pertahankan gaya hidup sehat, batasi asupan lemak jenuh, dan rutin lakukan aktivitas fisik ringan minimal 150 menit per minggu.")

st.write("")
st.write("")
st.markdown("<p style='text-align: center; color:#64748b; font-size:12px;'>Aplikasi Web Luaran Proyek - Universitas 17 Agustus 1945 Surabaya © 2026</p>", unsafe_allow_html=True)
