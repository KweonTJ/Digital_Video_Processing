import cv2
import numpy as np

def imread_korean(path, flag=cv2.IMREAD_COLOR):
    data = np.fromfile(path, np.uint8)
    return cv2.imdecode(data, flag)

# 이미지 경로 (한글 포함)
img_path = r"C:\Users\user\Desktop\2학기\디지털 영상 처리\chap09\cannytest.png"
img = imread_korean(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny_test(ksize, low_t, high_t):
    blur = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    edges = cv2.Canny(blur, low_t, high_t)
    return edges

edges1 = canny_test(3, 50, 150)
edges2 = canny_test(5, 50, 150)
edges3 = canny_test(7, 50, 150)

cv2.imshow("ksize3_50_150", edges1)
cv2.imshow("ksize5_50_150", edges2)
cv2.imshow("ksize7_50_150", edges3)
cv2.waitKey(0)
cv2.destroyAllWindows()
