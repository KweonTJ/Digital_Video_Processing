import cv2
from pathlib import Path
import numpy as np

img_path = r"C:\Users\user\Downloads\color.jpg"

img_bgr = cv2.imread(img_path)
if img_bgr is None:
    raise Exception("파일 읽기 오류")

# OpenCV는 문자열 경로만 받음
img_bgr = cv2.imread(str(img_path))
if img_bgr is None:
    raise Exception("파일 읽기 오류")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
r, g, b = cv2.split(img_rgb)

# 변환
r_prime = cv2.flip(r, 0)   # R: 상하 반전
g_prime = cv2.flip(g, 1)   # G: 좌우 반전
b_t     = cv2.transpose(b) # B: 전치

merged_rgb = cv2.merge([r_prime, g_prime, b])

# 출력
cv2.imshow("R' (up-down flipped)", r_prime)
cv2.imshow("G' (left-right flipped)", g_prime)
cv2.imshow("B (transpose)", b_t)
cv2.imshow("Merged RGB (R', G', B)", merged_rgb)
cv2.imshow("Original RGB", img_rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()
