import cv2
import numpy as np

def apply_clahe(image_rgb, clip_limit, grid_size=(8,8)):
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    
    l = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    return final_img

def apply_nlm(image_rgb, h_strength):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    denoised_bgr = cv2.fastNlMeansDenoisingColored(
        image_bgr,
        None,
        h=h_strength,
        hColor=h_strength,
        templateWindowSize=7,
        searchWindowSize=21
    )

    denoised_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
    return denoised_rgb