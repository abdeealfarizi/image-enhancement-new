import numpy as np
import cv2
import math
from collections import OrderedDict
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

_PC_CACHE = OrderedDict()
_PC_CACHE_MAXSIZE = 4

def _phase_congruency_cached(img):
    """
    Membungkus _phase_congruency dengan cache kecil (maksimal 4 entri).
    Mencegah komputasi FFT yang mahal dihitung ulang saat citra yang sama
    (mis. citra asli) dibandingkan lebih dari satu kali secara berurutan,
    seperti pada app.py: calculate_metrics(original, clahe_res) lalu
    calculate_metrics(original, nlm_res).
    """
    key = (img.shape, img.tobytes())
    if key in _PC_CACHE:
        _PC_CACHE.move_to_end(key)
        return _PC_CACHE[key]

    pc = _phase_congruency(img)
    _PC_CACHE[key] = pc
    if len(_PC_CACHE) > _PC_CACHE_MAXSIZE:
        _PC_CACHE.popitem(last=False)
    return pc

def _lowpass_filter(size, cutoff, n):
    rows, cols = size
    x = (np.arange(cols) - np.floor(cols / 2)) / cols
    y = (np.arange(rows) - np.floor(rows / 2)) / rows
    xx, yy = np.meshgrid(x, y)
    radius = np.sqrt(xx ** 2 + yy ** 2)
    return np.fft.ifftshift(1.0 / (1.0 + (radius / cutoff) ** (2 * n)))

def _phase_congruency(img):
    nscale = 4
    norient = 4
    min_wavelength = 6.0
    mult = 2.0
    sigma_onf = 0.55
    cutoff = 0.5
    g = 10.0
    epsilon = 1e-4

    rows, cols = img.shape
    imagefft = np.fft.fft2(img)

    x = (np.arange(cols) - np.floor(cols / 2)) / cols
    y = (np.arange(rows) - np.floor(rows / 2)) / rows
    xx, yy = np.meshgrid(x, y)
    radius = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(-yy, xx)
    radius = np.fft.ifftshift(radius)
    theta = np.fft.ifftshift(theta)
    radius[0, 0] = 1.0

    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    lp = _lowpass_filter((rows, cols), cutoff, 15)

    log_gabor = []
    for s in range(nscale):
        wavelength = min_wavelength * (mult ** s)
        fo = 1.0 / wavelength
        lg = np.exp(-(np.log(radius / fo)) ** 2 / (2 * np.log(sigma_onf) ** 2))
        lg *= lp
        lg[0, 0] = 0.0
        log_gabor.append(lg)

    energy_total = np.zeros((rows, cols))

    for o in range(norient):
        angle_o = o * np.pi / norient
        ds = sintheta * np.cos(angle_o) - costheta * np.sin(angle_o)
        dc = costheta * np.cos(angle_o) + sintheta * np.sin(angle_o)
        dtheta = np.abs(np.arctan2(ds, dc))
        dtheta = np.minimum(dtheta * norient / 2.0, np.pi)
        spread = (np.cos(dtheta) + 1.0) / 2.0

        sum_e_o = np.zeros((rows, cols))
        sum_o_o = np.zeros((rows, cols))
        sum_an_o = np.zeros((rows, cols))
        an_array = []

        for s in range(nscale):
            filt = log_gabor[s] * spread
            eo = np.fft.ifft2(imagefft * filt)
            even, odd = np.real(eo), np.imag(eo)
            an = np.sqrt(even ** 2 + odd ** 2)
            an_array.append(an)
            sum_e_o += even
            sum_o_o += odd
            sum_an_o += an

        median_e2n = float(np.median(an_array[0] ** 2))
        mean_e2n = median_e2n / np.log(2.0)
        noise_power = mean_e2n / max(rows * cols, 1)

        sum_an2 = sum(float(np.sum(a ** 2)) for a in an_array)
        sum_ai_aj = 0.0
        for i in range(nscale):
            for j in range(i + 1, nscale):
                sum_ai_aj += float(np.sum(an_array[i] * an_array[j]))
        est_noise_energy2 = sum_an2 * noise_power + 2 * noise_power * sum_ai_aj
        tau = np.sqrt(max(est_noise_energy2, 0.0) / max(rows * cols, 1))
        t = max(tau * np.sqrt(np.pi / 2.0), epsilon)

        energy = np.sqrt(sum_e_o ** 2 + sum_o_o ** 2)
        width = (sum_an_o / (an_array[0] + epsilon)) / nscale
        weight = 1.0 / (1.0 + np.exp((cutoff - width) * g))

        pc_o = weight * np.maximum(energy - t, 0.0) / (sum_an_o + epsilon)
        energy_total += pc_o

    pc = energy_total / norient
    return np.clip(pc, 0.0, 1.0)

def _fsim(gray_a, gray_b):
    T1 = 0.85
    T2 = 160.0
    
    pc1 = _phase_congruency_cached(gray_a)
    pc2 = _phase_congruency_cached(gray_b)
    
    gx1 = cv2.Scharr(gray_a, cv2.CV_64F, 1, 0)
    gy1 = cv2.Scharr(gray_a, cv2.CV_64F, 0, 1)
    gm1 = cv2.magnitude(gx1, gy1)
    
    gx2 = cv2.Scharr(gray_b, cv2.CV_64F, 1, 0)
    gy2 = cv2.Scharr(gray_b, cv2.CV_64F, 0, 1)
    gm2 = cv2.magnitude(gx2, gy2)
    
    s_pc = (2 * pc1 * pc2 + T1) / (pc1**2 + pc2**2 + T1)
    s_g = (2 * gm1 * gm2 + T2) / (gm1**2 + gm2**2 + T2)
    
    pc_m = np.maximum(pc1, pc2)
    
    fsim_val = np.sum(s_pc * s_g * pc_m) / (np.sum(pc_m) + 1e-8)
    
    if np.isnan(fsim_val) or np.isinf(fsim_val):
        return 0.0
        
    return float(fsim_val)

def calculate_metrics(image_a, image_b):
    if len(image_a.shape) == 3:
        img_a_gray = cv2.cvtColor(image_a, cv2.COLOR_RGB2GRAY)
        img_b_gray = cv2.cvtColor(image_b, cv2.COLOR_RGB2GRAY)
    else:
        img_a_gray = image_a
        img_b_gray = image_b

    try:
        score_mse = mse(image_a, image_b)
        score_rmse = math.sqrt(score_mse)
        score_psnr = psnr(image_a, image_b, data_range=255)
    except Exception:
        score_mse = 0.0
        score_rmse = 0.0
        score_psnr = 0.0
    
    min_dim = min(img_a_gray.shape[:2])
    win_size = 7
    if min_dim < 7:
        win_size = min_dim if min_dim % 2 != 0 else min_dim - 1
        if win_size < 3:
            win_size = 3 

    try:
        score_ssim = ssim(img_a_gray, img_b_gray, data_range=255.0, win_size=win_size)
        if np.isnan(score_ssim) or np.isinf(score_ssim):
            score_ssim = 0.0
    except ValueError:
        score_ssim = 0.0 

    try:
        score_fsim = _fsim(img_a_gray.astype(np.float64), img_b_gray.astype(np.float64))
    except Exception:
        score_fsim = 0.0

    return {
        "MSE": round(score_mse, 4),
        "RMSE": round(score_rmse, 4),
        "PSNR": round(score_psnr, 4),
        "SSIM": round(score_ssim, 4),
        "FSIM": round(score_fsim, 4)
    }