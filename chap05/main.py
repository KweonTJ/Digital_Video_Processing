# import numpy as np, cv2, math
#
# def calc_his(bgr):
#     B, G, R = float(bgr[0]), float(bgr[1]), float(bgr[2])
#     bgr_sum = (R + G + B)
#     tmp1 = ((R-G)+(R - B)) * 0.5
#     tmp2 = math.sqrt((R-G)*(R-G)+(R-B)*(G-B))
#     angle = math.acos(tmp1/ tmp2) * (180/np.pi) if tmp2 else 0
#
#     H = angle if B <= G else 360 - angle
#     S = 1.0 - 3 * min([R, G, B]) / bgr_sum if bgr_sum else 0
#     I = bgr_sum / 3
#     return (H/2, S*255, I)
#
# def bgr2hsi(image):
#     hsv = [[calc_his(pixel) for pixel in row] for row in image]
#     return cv2.convertScaleAbs(np.array(hsv))
#
# BGR_img = cv2.imread(r"C:\Users\user\Desktop\2학기\디지털 영상 처리\chap05\filter_blur.jpg", cv2.IMREAD_COLOR)
# if BGR_img is None: raise Exception("영상파일 읽기 오류")
#
# HSI_img = bgr2hsi(BGR_img)
# HSV_img = cv2.cvtColor(HSI_img, cv2.COLOR_BGR2HSV)
# Hue, Saturation, Intensity = cv2.split(HSV_img)
# Hue2, Saturation2, Intensity2 = cv2.split(HSV_img)
#
# titels = ['BGR_img', 'Hlue', 'Saturation', 'Intensity']
# [cv2.imshow(t, eval(t)) for t in titels]
# [cv2.imshow('OpenCV_'+t, eval(t+'2'))for t in titels[1:]]
# cv2.waitKey(0)
#
#
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 이미지 읽기 및 정규화
# image = cv2.imread('../data/Lena.png').astype(np.float32) / 255
#
# # 노이즈 추가
# noised = (image + 0.2 * np.random.rand(*image.shape).astype(np.float32))
# noised = noised.clip(0, 1)
#
# # 원본 + 노이즈 이미지 출력
# plt.imshow(noised[:, :, [2, 1, 0]])
# plt.show()
#
# # 1. Gaussian Blur
# gauss_blur = cv2.GaussianBlur(noised, (7, 7), 0)
# plt.imshow(gauss_blur[:, :, [2, 1, 0]])
# plt.show()
#
# # 2. Median Blur
# median_blur = cv2.medianBlur((noised * 255).astype(np.uint8), 7)
# plt.imshow(median_blur[:, :, [2, 1, 0]])
# plt.show()
#
# # 3. Bilateral Filter
# bilat = cv2.bilateralFilter(noised, -1, 0.3, 10)
# plt.imshow(bilat[:, :, [2, 1, 0]])
# plt.show()

# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

PATH_BLUR    = r"C:\Users\user\Desktop\2학기\디지털 영상 처리\chap05\filter_blur.jpg"
PATH_SHARPEN = r"C:\Users\user\Desktop\2학기\디지털 영상 처리\chap05\filter_sharpen.jpg"

def imread_u(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(path, np.uint8)
    return cv2.imdecode(data, flags)

def bgr_to_hsi(img_bgr):
    bgr = img_bgr.astype(np.float32) / 255.0
    B, G, R = bgr[...,0], bgr[...,1], bgr[...,2]
    eps = 1e-7
    num = 0.5*((R-G)+(R-B))
    den = np.sqrt((R-G)**2 + (R-B)*(G-B)) + eps
    theta = np.degrees(np.arccos(np.clip(num/den, -1.0, 1.0)))
    H = np.where(B <= G, theta, 360.0 - theta)
    S = 1.0 - 3.0*np.minimum(np.minimum(R,G),B)/(R+G+B + eps)
    I = (R+G+B)/3.0
    return np.stack([H, S, I], axis=-1).astype(np.float32)

def hsi_to_bgr(hsi):
    H, S, I = hsi[...,0], np.clip(hsi[...,1], 0, 1), np.clip(hsi[...,2], 0, 1)
    R = np.zeros_like(I); G = np.zeros_like(I); B = np.zeros_like(I)
    eps = 1e-7
    mask = (H >= 0) & (H < 120)
    h = np.radians(H[mask])
    B[mask] = I[mask]*(1 - S[mask])
    R[mask] = I[mask]*(1 + (S[mask]*np.cos(h))/(np.cos(np.radians(60) - h)+eps))
    G[mask] = 3*I[mask] - (R[mask] + B[mask])
    mask = (H >= 120) & (H < 240)
    h = np.radians(H[mask]-120)
    R[mask] = I[mask]*(1 - S[mask])
    G[mask] = I[mask]*(1 + (S[mask]*np.cos(h))/(np.cos(np.radians(60)-h)+eps))
    B[mask] = 3*I[mask] - (R[mask] + G[mask])
    mask = (H >= 240) & (H < 360)
    h = np.radians(H[mask]-240)
    G[mask] = I[mask]*(1 - S[mask])
    B[mask] = I[mask]*(1 + (S[mask]*np.cos(h))/(np.cos(np.radians(60)-h)+eps))
    R[mask] = 3*I[mask] - (G[mask] + B[mask])
    return (np.clip(np.stack([B,G,R],axis=-1),0,1)*255).astype(np.uint8)

def process_hsi_filtering(path):
    img = imread_u(path)
    if img is None:
        raise FileNotFoundError("filter_blur.jpg 불러오기 실패")
    hsi = bgr_to_hsi(img)
    H,S,I = hsi[...,0],hsi[...,1],hsi[...,2]
    H_g = cv2.GaussianBlur(H,(7,7),1.2)
    S_m = cv2.medianBlur((S*255).astype(np.uint8),5).astype(np.float32)/255.0
    I_min = cv2.erode((I*255).astype(np.uint8),np.ones((5,5),np.uint8)).astype(np.float32)/255.0
    bgr_rec = hsi_to_bgr(np.stack([H_g,S_m,I_min],axis=-1))
    return img, bgr_rec, H, S, I, H_g, S_m, I_min

def process_lab_bilateral(path):
    img = imread_u(path)
    if img is None:
        raise FileNotFoundError("filter_sharpen.jpg 불러오기 실패")
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L,a,b = cv2.split(lab)
    Lf = cv2.bilateralFilter(L, d=9, sigmaColor=40, sigmaSpace=15)
    out = cv2.merge([Lf,a,b])
    out_bgr = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
    return img, out_bgr

if __name__ == "__main__":
    blur_orig, blur_filtered, H, S, I, H_g, S_m, I_min = process_hsi_filtering(PATH_BLUR)
    sharp_orig, sharp_filtered = process_lab_bilateral(PATH_SHARPEN)

    # ----- 모든 결과를 한 번에 표시 -----
    plt.figure(figsize=(14,10))

    # filter_blur 관련
    plt.subplot(3,3,1); plt.imshow(cv2.cvtColor(blur_orig, cv2.COLOR_BGR2RGB)); plt.title("filter_blur - Original"); plt.axis('off')
    plt.subplot(3,3,2); plt.imshow(H, cmap='hsv'); plt.title("H (before)"); plt.axis('off')
    plt.subplot(3,3,3); plt.imshow(H_g, cmap='hsv'); plt.title("H after Gaussian"); plt.axis('off')
    plt.subplot(3,3,4); plt.imshow(S, cmap='gray'); plt.title("S (before)"); plt.axis('off')
    plt.subplot(3,3,5); plt.imshow(S_m, cmap='gray'); plt.title("S after Median"); plt.axis('off')
    plt.subplot(3,3,6); plt.imshow(I_min, cmap='gray'); plt.title("I after Min"); plt.axis('off')
    plt.subplot(3,3,7); plt.imshow(cv2.cvtColor(blur_filtered, cv2.COLOR_BGR2RGB)); plt.title("HSI Filtered RGB"); plt.axis('off')

    # filter_sharpen 관련
    plt.subplot(3,3,8); plt.imshow(cv2.cvtColor(sharp_orig, cv2.COLOR_BGR2RGB)); plt.title("filter_sharpen - Original"); plt.axis('off')
    plt.subplot(3,3,9); plt.imshow(cv2.cvtColor(sharp_filtered, cv2.COLOR_BGR2RGB)); plt.title("Lab Bilateral (L)"); plt.axis('off')

    plt.tight_layout()
    plt.show()
