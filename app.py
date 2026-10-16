import streamlit as st
import pandas as pd
import cv2
import io
import os
import sys
import altair as alt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.processing import apply_clahe, apply_nlm, apply_nlm_clahe
from src.metrics import calculate_metrics

st.set_page_config(
    page_title="Image Enhancement",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

URUTAN_METRIK = ["MSE", "RMSE", "PSNR", "SSIM", "FSIM"]

def _kolom_log_kosong():
    cols = pd.MultiIndex.from_tuples([
        ("CITRA", "Nama File"), ("PARAMETER", "Clip Limit"), ("PARAMETER", "Ukuran Grid"), ("PARAMETER", "Kekuatan NLM (h)"),
        ("CLAHE", "MSE"), ("CLAHE", "RMSE"), ("CLAHE", "PSNR"), ("CLAHE", "SSIM"), ("CLAHE", "FSIM"),
        ("CLAHE-NLM", "MSE"), ("CLAHE-NLM", "RMSE"), ("CLAHE-NLM", "PSNR"), ("CLAHE-NLM", "SSIM"), ("CLAHE-NLM", "FSIM"),
        ("NLM-CLAHE", "MSE"), ("NLM-CLAHE", "RMSE"), ("NLM-CLAHE", "PSNR"), ("NLM-CLAHE", "SSIM"), ("NLM-CLAHE", "FSIM")
    ])
    return pd.DataFrame(columns=cols)

if 'data_log' not in st.session_state:
    st.session_state['data_log'] = _kolom_log_kosong()

if 'image_gallery' not in st.session_state:
    st.session_state['image_gallery'] = []

if 'file_uploader_key' not in st.session_state:
    st.session_state['file_uploader_key'] = 0

def reset_data():
    st.session_state['data_log'] = _kolom_log_kosong()
    st.session_state['image_gallery'] = []
    st.session_state['file_uploader_key'] += 1
    st.session_state['tampilkan_toast_reset'] = True
    st.rerun()

if st.session_state.get('tampilkan_toast_reset', False):
    st.toast("Data berhasil direset.", icon=":material/check_circle:")
    st.session_state['tampilkan_toast_reset'] = False

if st.session_state.get('tampilkan_toast_selesai', False):
    st.toast("Proses selesai.", icon=":material/check_circle:")
    st.session_state['tampilkan_toast_selesai'] = False

@st.cache_data(show_spinner=False)
def generate_excel_buffer(data_log_df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        data_log_df.to_excel(writer, index=True, sheet_name='Log Eksperimen Lengkap')
    return buffer.getvalue()

def plot_histogram_interactive(image, title_label):
    df_hist = pd.DataFrame()
    saluran_warna = ('Merah', 'Hijau', 'Biru')
    warna_saluran = {'Merah': '#E2604F', 'Hijau': '#6FBF8B', 'Biru': '#6E93BE'}
    
    for nama_warna, i in zip(saluran_warna, range(3)):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        hist_data = hist.flatten()
        temp_df = pd.DataFrame({
            'Nilai Piksel': range(256),
            'Frekuensi': hist_data,
            'Saluran': nama_warna
        })
        df_hist = pd.concat([df_hist, temp_df], ignore_index=True)

    chart = alt.Chart(df_hist).mark_line(opacity=0.9, strokeWidth=2.0).encode(
        x=alt.X('Nilai Piksel', title='Nilai Piksel'),
        y=alt.Y('Frekuensi', title='Frekuensi'),
        color=alt.Color('Saluran', scale=alt.Scale(domain=list(saluran_warna), range=[warna_saluran[s] for s in saluran_warna]), legend=alt.Legend(title="Saluran")),
        tooltip=['Saluran', 'Nilai Piksel', 'Frekuensi']
    ).properties(
        title=title_label,
        height=260,
        background='transparent'
    ).configure_axis(
        grid=False,
        domainColor='#3A3B31',
        tickColor='#3A3B31',
        labelColor='#9A9D8E',
        titleColor='#9A9D8E',
        labelFontSize=10
    ).configure_view(
        strokeWidth=0
    ).configure_title(
        color='#F2EFE6',
        fontSize=13,
        anchor='middle',
        dy=-10
    ).interactive()

    return chart

@st.fragment
def render_galeri(gallery_data):
    search_query = st.text_input(
        "Cari file",
        value="",
        placeholder="Cari nama file...",
        icon=":material/search:",
        label_visibility="collapsed",
    )

    first_shown = True
    jumlah_cocok = 0
    
    for idx, item in enumerate(gallery_data, start=1):
        if search_query and search_query.lower() not in item["filename"].lower():
            continue
        jumlah_cocok += 1
        with st.expander(f"Citra #{idx} — {item['filename']}", icon=":material/image:", expanded=first_shown):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.image(item["original"], caption="Citra Asli", use_container_width=True)
            with c2:
                st.image(item["clahe"], caption="CLAHE", use_container_width=True)
            with c3:
                st.image(item["nlm"], caption="CLAHE-NLM", use_container_width=True)
            with c4:
                st.image(item["nlm_clahe"], caption="NLM-CLAHE", use_container_width=True)

            m_clahe = item["metrics_clahe"]
            m_hybrid = item["metrics_hybrid"]
            m_nlm_clahe = item["metrics_nlm_clahe"]

            st.caption("Metrik CLAHE")
            cols_clahe = st.columns(5)
            for mcol, nama_metrik in zip(cols_clahe, URUTAN_METRIK):
                mcol.metric(nama_metrik, f"{m_clahe[nama_metrik]:.4f}", border=True)

            st.caption("Metrik CLAHE-NLM")
            cols_hybrid = st.columns(5)
            for mcol, nama_metrik in zip(cols_hybrid, URUTAN_METRIK):
                mcol.metric(nama_metrik, f"{m_hybrid[nama_metrik]:.4f}", border=True)

            st.caption("Metrik NLM-CLAHE")
            cols_nlm_clahe = st.columns(5)
            for mcol, nama_metrik in zip(cols_nlm_clahe, URUTAN_METRIK):
                mcol.metric(nama_metrik, f"{m_nlm_clahe[nama_metrik]:.4f}", border=True)

            if st.checkbox("Tampilkan Histogram", key=f"hist_{item['filename']}_{idx}"):
                hc1, hc2, hc3, hc4 = st.columns(4)
                with hc1:
                    st.altair_chart(plot_histogram_interactive(item["original"], "Citra Asli"), use_container_width=True)
                with hc2:
                    st.altair_chart(plot_histogram_interactive(item["clahe"], "CLAHE"), use_container_width=True)
                with hc3:
                    st.altair_chart(plot_histogram_interactive(item["nlm"], "CLAHE-NLM"), use_container_width=True)
                with hc4:
                    st.altair_chart(plot_histogram_interactive(item["nlm_clahe"], "NLM-CLAHE"), use_container_width=True)
        first_shown = False

    if search_query and jumlah_cocok == 0:
        st.warning("File tidak ditemukan.", icon=":material/search_off:")

with st.sidebar:
    st.subheader(":material/upload: Sumber Citra")
    uploaded_files = st.file_uploader(
        "Unggah Citra",
        type=['jpg', 'jpeg'],
        accept_multiple_files=True,
        key=str(st.session_state['file_uploader_key'])
    )

    st.subheader(":material/tune: Parameter Algoritma")
    clip_limit = st.slider("Clip Limit (CLAHE)", 2.0, 4.0, 2.0, 0.1)
    st.caption("Ukuran tile grid konstan 8x8.")
    nlm_h = st.slider("Filter NLM (h)", 1, 20, 10, 1)

    run_btn = st.button(
        "Proses",
        type="primary",
        icon=":material/play_arrow:",
        use_container_width=True,
    )

    if st.button("Atur Ulang", icon=":material/restart_alt:", use_container_width=True):
        reset_data()

#logo
col1, col2 = st.columns([1, 15], vertical_alignment="center")

with col1:
    st.image("logo.png", width=60)

with col2:
    st.markdown("# Image Enhancement")
    st.caption("Low-light Image Enhancement Using CLAHE and CLAHE-NLM")
#logo

if run_btn:
    if uploaded_files:
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_files = len(uploaded_files)

        rows_baru = []

        for i, file in enumerate(uploaded_files):
            status_text.text(f"Memproses {i+1}/{total_files}: {file.name}")
            
            if not st.session_state['data_log'].empty and file.name in st.session_state['data_log'][("CITRA", "Nama File")].values:
                st.warning(f"Citra dengan nama {file.name} sudah pernah diproses sebelumnya. Melewati citra ini.", icon="⚠️")
                continue

            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            original_img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

            if original_img is None:
                st.warning(f"File {file.name} korup atau tidak valid. Melewati citra ini.", icon="⚠️")
                continue

            if original_img.ndim != 3 or original_img.shape[2] != 3:
                st.warning(f"Citra {file.name} tidak memiliki tiga saluran warna (RGB) yang valid. Melewati citra ini.", icon="⚠️")
                continue

            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

            clahe_res = apply_clahe(original_img, clip_limit, (8,8))
            nlm_res = apply_nlm(clahe_res, nlm_h)
            nlm_clahe_res = apply_nlm_clahe(original_img, clip_limit, nlm_h, (8,8))

            metrics_clahe = calculate_metrics(original_img, clahe_res)
            metrics_hybrid = calculate_metrics(original_img, nlm_res)
            metrics_nlm_clahe = calculate_metrics(original_img, nlm_clahe_res)

            new_row = {
                ("CITRA", "Nama File"): file.name,
                ("PARAMETER", "Clip Limit"): clip_limit,
                ("PARAMETER", "Ukuran Grid"): "8x8",
                ("PARAMETER", "Kekuatan NLM (h)"): nlm_h,
                ("CLAHE", "MSE"): metrics_clahe["MSE"],
                ("CLAHE", "RMSE"): metrics_clahe["RMSE"],
                ("CLAHE", "PSNR"): metrics_clahe["PSNR"],
                ("CLAHE", "SSIM"): metrics_clahe["SSIM"],
                ("CLAHE", "FSIM"): metrics_clahe["FSIM"],
                ("CLAHE-NLM", "MSE"): metrics_hybrid["MSE"],
                ("CLAHE-NLM", "RMSE"): metrics_hybrid["RMSE"],
                ("CLAHE-NLM", "PSNR"): metrics_hybrid["PSNR"],
                ("CLAHE-NLM", "SSIM"): metrics_hybrid["SSIM"],
                ("CLAHE-NLM", "FSIM"): metrics_hybrid["FSIM"],
                ("NLM-CLAHE", "MSE"): metrics_nlm_clahe["MSE"],
                ("NLM-CLAHE", "RMSE"): metrics_nlm_clahe["RMSE"],
                ("NLM-CLAHE", "PSNR"): metrics_nlm_clahe["PSNR"],
                ("NLM-CLAHE", "SSIM"): metrics_nlm_clahe["SSIM"],
                ("NLM-CLAHE", "FSIM"): metrics_nlm_clahe["FSIM"]
            }
            rows_baru.append(new_row)

            st.session_state['image_gallery'].append({
                "filename": file.name,
                "original": original_img,
                "clahe": clahe_res,
                "nlm": nlm_res,
                "nlm_clahe": nlm_clahe_res,
                "metrics_clahe": metrics_clahe,
                "metrics_hybrid": metrics_hybrid,
                "metrics_nlm_clahe": metrics_nlm_clahe,
            })

            progress_bar.progress((i + 1) / total_files)

        if rows_baru:
            st.session_state['data_log'] = pd.concat(
                [st.session_state['data_log'], pd.DataFrame(rows_baru)], ignore_index=True
            )
            st.session_state['data_log'].index = range(1, len(st.session_state['data_log']) + 1)

        status_text.empty()
        progress_bar.empty()
        
        st.session_state['file_uploader_key'] += 1
        st.session_state['tampilkan_toast_selesai'] = True
        st.rerun()
    else:
        st.warning("Silahkan unggah minimal satu citra terlebih dahulu sebelum memproses.", icon="⚠️")

if st.session_state['image_gallery']:
    total_citra = len(st.session_state['image_gallery'])
    st.subheader(f"Hasil Proses ({total_citra} Citra)")
    
    render_galeri(st.session_state['image_gallery'])

    st.subheader("Log Data Akumulasi")
    st.dataframe(st.session_state['data_log'].style.format({
        ("CLAHE", "MSE"): "{:.4f}", ("CLAHE", "RMSE"): "{:.4f}", ("CLAHE", "PSNR"): "{:.4f}", ("CLAHE", "SSIM"): "{:.4f}", ("CLAHE", "FSIM"): "{:.4f}",
        ("CLAHE-NLM", "MSE"): "{:.4f}", ("CLAHE-NLM", "RMSE"): "{:.4f}", ("CLAHE-NLM", "PSNR"): "{:.4f}", ("CLAHE-NLM", "SSIM"): "{:.4f}", ("CLAHE-NLM", "FSIM"): "{:.4f}",
        ("NLM-CLAHE", "MSE"): "{:.4f}", ("NLM-CLAHE", "RMSE"): "{:.4f}", ("NLM-CLAHE", "PSNR"): "{:.4f}", ("NLM-CLAHE", "SSIM"): "{:.4f}", ("NLM-CLAHE", "FSIM"): "{:.4f}"
    }), use_container_width=True)

    excel_data = generate_excel_buffer(st.session_state['data_log'])
    st.download_button(
        "Unduh Excel",
        data=excel_data,
        file_name="Laporan_Hasil_Proses.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        icon=":material/download:",
    )
elif uploaded_files:
    st.info(f"{len(uploaded_files)} citra siap diproses. Klik 'Proses' di panel kiri untuk memulai.", icon=":material/info:")
else:
    st.info("Silahkan unggah citra.", icon=":material/upload_file:")