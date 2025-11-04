# main.py  (한글 경로 대응 완전판)
import cv2 as cv
import numpy as np
from pathlib import Path
import sys

# 한글 포함 절대경로
IMG_PATH = Path(r"C:\Users\user\Desktop\2학기\디지털 영상 처리\chap07\dft_300.jpg")

def imread_unicode(path, flag=cv.IMREAD_GRAYSCALE):
    """한글 경로에서도 읽히는 안전한 이미지 로더"""
    try:
        data = np.fromfile(str(path), np.uint8)  # 한글경로용
        img = cv.imdecode(data, flag)
        if img is None:
            raise ValueError("imdecode 실패")
        return img
    except Exception as e:
        print("이미지 로드 실패:", e)
        print("경로:", path)
        sys.exit(1)

def optimal_pad(img):
    h, w = img.shape
    H = cv.getOptimalDFTSize(h)
    W = cv.getOptimalDFTSize(w)
    padded = cv.copyMakeBorder(img, 0, H-h, 0, W-w, cv.BORDER_CONSTANT, value=0)
    return padded, (h, w)

def dft_shift(img32):
    dft = cv.dft(img32, flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft, axes=(0, 1))
    return dft_shifted

def idft_shift(dft_shifted, out_size):
    dft_ishift = np.fft.ifftshift(dft_shifted, axes=(0, 1))
    img_back_complex = cv.idft(dft_ishift)
    img_back = cv.magnitude(img_back_complex[:,:,0], img_back_complex[:,:,1])
    img_back = img_back[:out_size[0], :out_size[1]]
    img_back = cv.normalize(img_back, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    return img_back

def log_magnitude(dft_or_shifted):
    mag = cv.magnitude(dft_or_shifted[:,:,0], dft_or_shifted[:,:,1])
    mag = np.log(mag + 1.0)
    mag = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    return mag

def make_mask(shape, size, pass_type, H, W):
    cx, cy = W // 2, H // 2
    mask = np.zeros((H, W), np.float32)
    if shape == 'square':
        half = int(round(size / 2))
        x1, x2 = cx - half, cx + half
        y1, y2 = cy - half, cy + half
        mask[y1:y2, x1:x2] = 1
    else:
        r = int(round(size / 2))
        yy, xx = np.ogrid[:H, :W]
        mask[((yy-cy)**2 + (xx-cx)**2) <= r*r] = 1
    if pass_type == 'high':
        mask = 1 - mask
    return mask, cv.merge([mask, mask])

def show_scaled(title, img, width=512):
    h, w = img.shape[:2]
    scale = width / w
    out = cv.resize(img, (int(w*scale), int(h*scale)))

    cv.namedWindow(title, cv.WINDOW_AUTOSIZE)
    cv.setWindowProperty(title, cv.WND_PROP_TOPMOST, 1)
    cv.imshow(title, out)

def main():
    img = imread_unicode(IMG_PATH)
    padded, orig_size = optimal_pad(img)
    img32 = padded.astype(np.float32)

    dft_shifted = dft_shift(img32)

    print("[필터 설정]")
    shape = input("형태 (square/circle) : ").strip().lower() or "square"
    if shape not in ("square", "circle"): shape = "square"
    try:
        size = int(input("크기(정사각형 변 & 원 지름 px) : ") or "50")
    except: size = 50
    pass_type = input("종류 (low/high) : ").strip().lower() or "low"
    if pass_type not in ("low", "high"): pass_type = "low"

    H, W = padded.shape
    size = max(1, min(size, min(H, W)))

    mask_vis, mask2 = make_mask(shape, size, pass_type, H, W)
    dft_filtered = dft_shifted * mask2

    mag_before = log_magnitude(dft_shifted)
    mag_after  = log_magnitude(dft_filtered)
    img_idft   = idft_shift(dft_filtered, orig_size)

    show_scaled("Main Video", img)
    show_scaled("Log magnitude", mag_before)
    show_scaled("Filter Video", mag_after)
    show_scaled("IDFT", img_idft)

    print("아무 키나 누르면 종료")
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
