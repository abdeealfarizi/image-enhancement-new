"""
Pilot test: bandingkan 3 skenario pemrosesan pada 1 citra uji.
  1. CLAHE saja
  2. CLAHE -> NLM   (pipeline utama skripsi, urutan sekarang)
  3. NLM -> CLAHE   (skenario baru, urutan dibalik, usulan Pak Henhen)

Semua dibandingkan terhadap citra ASLI (original) pakai 5 metrik:
MSE, RMSE, PSNR, SSIM, FSIM.

Cara pakai:
1. Taruh skrip ini di folder root project (sejajar sama app.py)
2. Ganti NAMA_FILE_CITRA di bawah sesuai nama file citra uji
   (ambil salah satu dari 40 sampel di sampel_terpilih.csv)
3. Jalankan: python pilot_test_nlm_clahe.py
"""

import sys
import os
import cv2
import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.processing import apply_clahe, apply_nlm, apply_nlm_clahe
from src.metrics import calculate_metrics

# ======== KONFIGURASI (samakan dengan parameter di app.py) ========
NAMA_FILE_CITRA = "dataset/2015_00024.JPG"   # GANTI sesuai nama file citra uji kamu
CLIP_LIMIT = 2.0
GRID_SIZE = (8, 8)
NLM_H = 10
# =====================================================================

def muat_citra(path):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Citra tidak ditemukan atau gagal dibaca: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

if __name__ == "__main__":
    original = muat_citra(NAMA_FILE_CITRA)

    # Skenario 1: CLAHE saja
    clahe_res = apply_clahe(original, CLIP_LIMIT, GRID_SIZE)

    # Skenario 2: CLAHE -> NLM (pipeline utama, urutan sekarang)
    clahe_nlm_res = apply_nlm(clahe_res, NLM_H)

    # Skenario 3: NLM -> CLAHE (urutan dibalik)
    nlm_clahe_res = apply_nlm_clahe(original, CLIP_LIMIT, NLM_H, GRID_SIZE)

    metrik_clahe = calculate_metrics(original, clahe_res)
    metrik_clahe_nlm = calculate_metrics(original, clahe_nlm_res)
    metrik_nlm_clahe = calculate_metrics(original, nlm_clahe_res)

    hasil = pd.DataFrame({
        "CLAHE": metrik_clahe,
        "CLAHE-NLM": metrik_clahe_nlm,
        "NLM-CLAHE": metrik_nlm_clahe,
    })

    print(f"\nCitra uji: {NAMA_FILE_CITRA}")
    print(hasil)

    # Selisih CLAHE-NLM vs NLM-CLAHE, biar langsung kelihatan beda signifikan atau tidak
    selisih = (hasil["NLM-CLAHE"] - hasil["CLAHE-NLM"]).round(4)
    print("\nSelisih (NLM-CLAHE - CLAHE-NLM):")
    print(selisih)

    hasil.to_csv("pilot_test_hasil.csv")
    print("\nTersimpan di: pilot_test_hasil.csv")
