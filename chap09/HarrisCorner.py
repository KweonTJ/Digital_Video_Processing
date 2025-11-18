import cv2
import numpy as np

def imread_korean(path, flag=cv2.IMREAD_COLOR):
    data = np.fromfile(path, np.uint8)
    return cv2.imdecode(data, flag)

def show_resized(title, img, scale=0.5):
    h, w = img.shape[:2]
    resized = cv2.resize(img, (int(w*scale), int(h*scale)))
    cv2.imshow(title, resized)

def harris_test(img, gray_f, block_size, k):
    dst = cv2.cornerHarris(gray_f, block_size, 3, k)
    dst = cv2.dilate(dst, None)

    result = img.copy()
    result[dst > 0.01 * dst.max()] = [0, 0, 255]
    return result

img_path = r"C:\Users\user\Desktop\2학기\디지털 영상 처리\chap09\harristest.jpg"
img = imread_korean(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_f = np.float32(gray)

h1 = harris_test(img, gray_f, 2, 0.04)
h2 = harris_test(img, gray_f, 5, 0.04)
h3 = harris_test(img, gray_f, 5, 0.1)

show_resized("block2_k004", h1, 0.5)
show_resized("block5_k004", h2, 0.5)
show_resized("block5_k01", h3, 0.5)

cv2.waitKey(0)
cv2.destroyAllWindows()
