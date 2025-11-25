# hough_circles_demo.py
import cv2
import numpy as np

def imread_korean(path, flag=cv2.IMREAD_COLOR):
    data = np.fromfile(path, np.uint8)
    return cv2.imdecode(data, flag)

# 1. 이미지 로드
circle_img_path = r"C:\Users\user\Desktop\2학기\디지털 영상 처리\chap10\HoughCircles.jpg"
circle_img = imread_korean(circle_img_path)

if circle_img is None:
    raise FileNotFoundError("HoughCircle 이미지 로드 실패.")

# 2. 전처리: Gray → medianBlur
gray = cv2.cvtColor(circle_img, cv2.COLOR_BGR2GRAY)
gray_blur = cv2.medianBlur(gray, 5)

cv2.imshow("Original", circle_img)
cv2.imshow("Gray + Blur", gray_blur)

# 3. 파라미터 세트(최소 3개)
param_sets = [
    { "name": "case1_dp1.2_p1=250_p2=40", "dp": 1.2, "minDist": 40, "param1": 300, "param2": 40, "minRadius": 0, "maxRadius": 0 },
    { "name": "case2_dp1.5_p1=270_p2=30", "dp": 1.5, "minDist": 40, "param1": 300, "param2": 30, "minRadius": 0, "maxRadius": 0 },
    { "name": "case3_dp1.0_p1=300_p2=50", "dp": 1.0, "minDist": 40, "param1": 300, "param2": 50, "minRadius": 0, "maxRadius": 0 },
]

# 4. HoughCircles 수행
for params in param_sets:
    dp        = params["dp"]
    minDist   = params["minDist"]
    param1    = params["param1"]
    param2    = params["param2"]
    minRadius = params["minRadius"]
    maxRadius = params["maxRadius"]

    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=param1,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius
    )

    result = cv2.cvtColor(gray_blur, cv2.COLOR_GRAY2BGR)
    num_circles = 0

    if circles is not None:
        circles = np.uint16(np.around(circles))
        num_circles = circles.shape[1]

        for (x, y, r) in circles[0, :]:
            cv2.circle(result, (x, y), r, (0, 255, 0), 2)
            cv2.circle(result, (x, y), 2, (0, 0, 255), 3)

    print(f"[{params['name']}] dp={dp}, param1={param1}, param2={param2}, count={num_circles}")
    cv2.imshow(f"HoughCircles_{params['name']}", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
