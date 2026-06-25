import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Dashboard Tingkat Kemiskinan",
    page_icon="🇮🇩",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = joblib.load('logistic_regression.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model_lr, scaler_lr = load_model()

st.title("Prediksi Tingkat Kemiskinan Wilayah")
st.markdown("""
Selamat datang di aplikasi prediksi status sosial ekonomi wilayah di Indonesia. 
Silakan masukkan indikator pada form di bawah untuk mengetahui apakah suatu wilayah diprediksi masuk dalam **Kategori 0 (Kemiskinan Rendah)** atau **Kategori 1 (Kemiskinan Tinggi)**.
""")

with st.form("form_prediksi"):
    st.subheader("Input Indikator Sosial Ekonomi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        reg_gdp = st.number_input(
            "PDRB / PDB Regional (Triliun Rp)", 
            min_value=0.0, value=34.0, format="%.2f",
            help="Produk Domestik Regional Bruto (PDRB) merupakan jumlah nilai tambah barang dan jasa yang dihasilkan dari seluruh kegiatan ekonomi di suatu wilayah."
        )
        life_exp = st.number_input(
            "Angka Harapan Hidup (Tahun)", 
            min_value=0.0, value=69.0, format="%.2f",
            help="Rata-rata perkiraan banyak tahun yang dapat ditempuh oleh seseorang selama hidup."
        )
        
    with col2:
        avg_schooltime = st.number_input(
            "Rata-rata Lama Sekolah (Tahun)", 
            min_value=0.0, max_value=20.0, value=8.5, format="%.2f",
            help="Jumlah tahun belajar penduduk usia 15 tahun ke atas yang telah diselesaikan dalam pendidikan formal."
        )
        exp_percap = st.number_input(
            "Pengeluaran Per Kapita (Ribu/Bulan)", 
            min_value=0, value=10000, step=100,
            help="Biaya yang dikeluarkan untuk konsumsi semua anggota rumah tangga selama sebulan dibagi dengan banyaknya anggota rumah tangga."
        )
        
    submitted = st.form_submit_button("Lakukan Prediksi")

if submitted:
    data_input = pd.DataFrame({
        'reg_gdp': [reg_gdp],
        'life_exp': [life_exp],
        'avg_schooltime': [avg_schooltime],
        'exp_percap': [exp_percap]
    })
    
    input_scaled = scaler_lr.transform(data_input)
    prediction = model_lr.predict(input_scaled)
    
    st.markdown("---")
    st.subheader("Hasil Klasifikasi Sistem")
    
    st.markdown("**Ringkasan Data Wilayah yang Diinput:**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PDRB", f"{reg_gdp} T")
    m2.metric("Harapan Hidup", f"{life_exp} Thn")
    m3.metric("Lama Sekolah", f"{avg_schooltime} Thn")
    m4.metric("Pengeluaran", f"Rp {exp_percap}k")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if prediction[0] == 0:
        st.success("**Kategori 0: Tingkat Kemiskinan Rendah**\n\nBerdasarkan indikator ekonomi dan pendidikan, wilayah ini diprediksi memiliki persentase penduduk miskin **di bawah** ambang batas nasional BPS (10,14%). Status ekonomi wilayah tergolong aman.")
        st.balloons() 
    elif prediction[0] == 1:
        st.error("**Kategori 1: Tingkat Kemiskinan Tinggi**\n\nIndikator di wilayah ini memerlukan intervensi program pengentasan kemiskinan lebih lanjut karena diprediksi memiliki persentase penduduk miskin **di atas atau sama dengan** ambang batas nasional BPS (10,14%).")

st.markdown("---")
st.markdown("*Dashboard ini dibangun menggunakan Model Logistic Regression dengan akurasi yang dievaluasi berdasarkan data BPS 2021.*")