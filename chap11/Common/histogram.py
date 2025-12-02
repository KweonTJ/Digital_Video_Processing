# Common/histogram.py
import numpy as np
import cv2

def draw_histo_hue(hist, size=(200, 256, 3)):
    """
    Hue 히스토그램(1차원 배열)을 막대 그래프 이미지로 그려서 반환.
    size: (height, width, channels)
    """
    h, w, _ = size
    hist_img = np.zeros(size, np.uint8)

    bins = len(hist)                # 보통 32
    bin_w = int(w / bins)

    # 히스토그램 값을 높이 범위(0~h-1)로 정규화
    hist_norm = cv2.normalize(hist, None, 0, h - 1, cv2.NORM_MINMAX)

    for i in range(bins):
        x1 = i * bin_w
        x2 = x1 + bin_w - 1
        y = h - int(hist_norm[i])
        cv2.rectangle(hist_img, (x1, y), (x2, h - 1), (255, 255, 255), -1)

    return hist_img
