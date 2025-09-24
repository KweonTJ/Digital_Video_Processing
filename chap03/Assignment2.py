import cv2
import numpy as np

# 이미지 경로
img1_path = r"C:\Users\user\Downloads\abs_test1.jpg"
img2_path = r"C:\Users\user\Downloads\abs_test2.jpg"

# 1. Grayscale로 읽기
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
if img1 is None or img2 is None:
    raise Exception("파일 읽기 오류")

x, y, w, h = 100, 100, 50, 50
roi1 = img1[y:y+h, x:x+w].astype(np.int16)  # int16로 변환하여 음수 대응
roi2 = img2[y:y+h, x:x+w].astype(np.int16)

# 차이 영상 계산
diff = roi1 - roi2

# 절댓값 영상 계산
abs_diff = np.abs(diff)

# 차이 영상 개선 (비율식 적용)
min_val = diff.min()
max_val = diff.max()
enhanced = ((diff - min_val) / (max_val - min_val) * 255).astype(np.uint8)

# 영상 출력
cv2.imshow("ROI1", roi1.astype(np.uint8))
cv2.imshow("ROI2", roi2.astype(np.uint8))
cv2.imshow("Difference", diff.astype(np.uint8))
cv2.imshow("Absolute Difference", abs_diff.astype(np.uint8))
cv2.imshow("Enhanced Difference", enhanced)

cv2.waitKey(0)
cv2.destroyAllWindows()