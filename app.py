import streamlit as st
import pandas as pd
import cv2
import io
import os
import sys
import altair as alt
import numpy as np

# Pancingan Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.processing import load_image, apply_clahe, apply_nlm
from src.metrics import calculate_metrics

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Analisis Peningkatan Citra Low-Light",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (Mode Ultra Gelap) ---
st.markdown("""
<style>
    /* Style Kartu Metrik */
    .stMetric {
        background-color: #121212; 
        border: 1px solid #333; 
        padding: 10px; 
        border-radius: 5px;
    }
    /* Style Judul */
    h1 {font-size: 28px !important; font-weight: 700; color: #FFFFFF;}
    h2, h3 {font-weight: 600; color: #E0E0E0;}
    
    /* Ukuran font tabel */
    .dataframe {font-size: 14px !important;} 
</style>
""", unsafe_allow_html=True)

# --- INISIALISASI SESSION STATE ---
if 'data_log' not in st.session_state:
    st.session_state['data_log'] = pd.DataFrame(columns=[
        "Nama File", "Clip Limit", "Ukuran Grid", "Kekuatan NLM (h)", 
        "MSE", "RMSE", "PSNR (dB)", "SSIM", "FSIM"
    ])

# --- FUNGSI RESET DATA ---
def reset_data():
    st.session_state['data_log'] = pd.DataFrame(columns=[
        "Nama File", "Clip Limit", "Ukuran Grid", "Kekuatan NLM (h)", 
        "MSE", "RMSE", "PSNR (dB)", "SSIM", "FSIM"
    ])
    st.success("Log data telah berhasil direset.")

# --- FUNGSI HISTOGRAM INTERAKTIF (ALTAIR) ---
def plot_histogram_interactive(image, title_label):
    """
    Membuat histogram interaktif menggunakan Altair dalam Bahasa Indonesia.
    """
    df_hist = pd.DataFrame()
    
    saluran_warna = ('Merah', 'Hijau', 'Biru')
    for i, nama_warna in enumerate(saluran_warna):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        hist_data = hist.flatten()
        
        temp_df = pd.DataFrame({
            'Nilai Piksel': range(256),
            'Frekuensi': hist_data,
            'Saluran': nama_warna
        })
        df_hist = pd.concat([df_hist, temp_df])

    chart = alt.Chart(df_hist).mark_line(opacity=0.9, strokeWidth=2.0).encode(
        x=alt.X('Nilai Piksel', title='Nilai Piksel'),
        y=alt.Y('Frekuensi', title='Frekuensi'),
        color=alt.Color('Saluran', scale=alt.Scale(domain=['Merah', 'Hijau', 'Biru'], range=['#FF4B4B', '#00CC96', '#0068C9']), legend=alt.Legend(title="Saluran")),
        tooltip=['Saluran', 'Nilai Piksel', 'Frekuensi']
    ).properties(
        title=title_label,
        height=300,
        background='transparent'
    ).configure_axis(
        grid=False,
        domainColor='#333',
        tickColor='#333',
        labelColor='#888',
        titleColor='#888',
        labelFontSize=10
    ).configure_view(
        strokeWidth=0
    ).configure_title(
        color='#E0E0E0',
        fontSize=14,
        anchor='middle',
        dy=-10
    ).interactive()
    
    return chart

# --- KONFIGURASI SIDEBAR ---
with st.sidebar:
    st.header("Konfigurasi Parameter")
    
    uploaded_file = st.file_uploader("Unggah Citra Uji", type=['png', 'jpg', 'jpeg'])
    
    st.markdown("---")
    st.subheader("Parameter Algoritma")
    
    clip_limit = st.slider("Ambang Batas Clip (CLAHE)", 2.0, 4.0, 2.0, 0.1)
    st.text_input("Ukuran Grid Tile", "8x8", disabled=True)
    nlm_h = st.slider("Kekuatan Filter NLM (h)", 1, 20, 10, 1)
    
    st.markdown("---")
    run_btn = st.button("PROSES CITRA", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Manajemen Data")
    if st.button("Reset Seluruh Log Data"):
        reset_data()

# --- HEADER UTAMA ---
st.title("Analisis Komparatif Peningkatan Citra Low-Light")
st.markdown("Perbandingan Metode: **CLAHE** vs **Hybrid CLAHE-NLM**")

# --- LOGIKA UTAMA ---
if uploaded_file:
    original_img = load_image(uploaded_file)
    file_name = uploaded_file.name
    
    if run_btn:
        with st.spinner("Memproses Algoritma & Menghitung Metrik..."):
            # 1. Pemrosesan
            clahe_res = apply_clahe(original_img, clip_limit, (8,8))
            nlm_res = apply_nlm(clahe_res, nlm_h)
            
            # 2. Perhitungan Metrik
            metrics = calculate_metrics(clahe_res, nlm_res)
            
            # 3. Simpan ke Session State
            new_row = {
                "Nama File": file_name,
                "Clip Limit": clip_limit,
                "Ukuran Grid": "8x8",
                "Kekuatan NLM (h)": nlm_h,
                "MSE": metrics["MSE"],
                "RMSE": metrics["RMSE"],
                "PSNR (dB)": metrics["PSNR"],
                "SSIM": metrics["SSIM"],
                "FSIM": metrics["FSIM"]
            }
            st.session_state['data_log'] = pd.concat([st.session_state['data_log'], pd.DataFrame([new_row])], ignore_index=True)
            
            # --- VISUALISASI (3 KOLOM) ---
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.image(original_img, caption="1. Citra Asli (Low-Light)", use_container_width=True)
                chart1 = plot_histogram_interactive(original_img, "Histogram Citra Asli")
                st.altair_chart(chart1, use_container_width=True)

            with c2:
                st.image(clahe_res, caption=f"2. Hasil CLAHE (Clip: {clip_limit})", use_container_width=True)
                chart2 = plot_histogram_interactive(clahe_res, "Histogram CLAHE")
                st.altair_chart(chart2, use_container_width=True)
                
            with c3:
                st.image(nlm_res, caption=f"3. Hasil CLAHE-NLM (h: {nlm_h})", use_container_width=True)
                chart3 = plot_histogram_interactive(nlm_res, "Histogram CLAHE-NLM")
                st.altair_chart(chart3, use_container_width=True)

    # --- TABEL LOG DATA ---
    st.markdown("---")
    st.subheader("Log Data Akumulasi")
    
    if not st.session_state['data_log'].empty:
        st.dataframe(st.session_state['data_log'].style.format({
            "MSE": "{:.4f}", "RMSE": "{:.4f}", "PSNR (dB)": "{:.4f}", "SSIM": "{:.4f}", "FSIM": "{:.4f}"
        }), use_container_width=True)
        
        # Tombol Ekspor
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            st.session_state['data_log'].to_excel(writer, index=False, sheet_name='Log Eksperimen Lengkap')
            
        st.download_button(
            label="Export Data",
            data=buffer.getvalue(),
            file_name="Laporan_Hasil_Eksperimen.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )
    else:
        st.info("Belum ada data yang tercatat. Klik 'PROSES CITRA' untuk menambahkan data.")

else:
    st.info("Silakan unggah gambar citra untuk memulai analisis.")