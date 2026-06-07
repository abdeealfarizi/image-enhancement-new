import cv2
import numpy as np

def load_image(image_file):
    """Membaca file gambar yang diupload user"""
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # Convert BGR (OpenCV standard) to RGB (Streamlit standard)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

def apply_clahe(image_rgb, clip_limit, grid_size=(8,8)):
    """
    Menerapkan algoritma CLAHE pada channel Luminance (LAB).
    Sesuai teori: CLAHE paling baik diterapkan di channel L pada Lab color space.
    """
    # 1. Convert RGB ke LAB
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # 2. Buat objek CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    
    # 3. Terapkan CLAHE hanya pada channel L (Lightness)
    cl = clahe.apply(l)
    
    # 4. Gabungkan kembali (Merge)
    limg = cv2.merge((cl,a,b))
    
    # 5. Convert balik ke RGB
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final_img

def apply_nlm(image_rgb, h_strength):
    """
    Menerapkan Non-Local Means Denoising.
    h_strength: Parameter kekuatan filter (semakin besar makin halus tapi makin blur).
    """
    # hColor biasanya disamakan dengan h (untuk citra berwarna)
    # templateWindowSize = 7 (default), searchWindowSize = 21 (default)
    denoised = cv2.fastNlMeansDenoisingColored(image_rgb, None, h=h_strength, hColor=h_strength, templateWindowSize=7, searchWindowSize=21)
    return denoised