# hough_lines_demo.py
import cv2
import numpy as np

def imread_korean(path, flag=cv2.IMREAD_COLOR):
    data = np.fromfile(path, np.uint8)
    return cv2.imdecode(data, flag)

# 1. 이미지 로드
img_path = r"C:\Users\user\Desktop\2학기\디지털 영상 처리\chap10\HoughLines.jpg"
img = imread_korean(img_path)

if img is None:
    raise FileNotFoundError("HoughLines 이미지 로드 실패.")

# 2. 전처리: Gray → Blur → Canny
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
edges = cv2.Canny(blur, 50, 150)

cv2.imshow("Original", img)
cv2.imshow("Edges", edges)

# 3. 파라미터 세트(최소 3개)
param_sets = [
    { "name": "case1_rho1_theta1_thr380", "rho": 1, "theta": np.pi/280, "threshold": 380 },
    { "name": "case2_rho1_theta1_thr375", "rho": 1, "theta": np.pi/180, "threshold": 377 },
    { "name": "case3_rho2_theta1_thr370", "rho": 2, "theta": np.pi/280, "threshold": 370 },
]

# 4. HoughLines 수행
for params in param_sets:
    rho    = params["rho"]
    theta  = params["theta"]
    thr    = params["threshold"]

    lines = cv2.HoughLines(edges, rho, theta, thr)
    result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    if lines is not None:
        for line in lines:
            rho_i, theta_i = line[0]
            a = np.cos(theta_i)
            b = np.sin(theta_i)
            x0 = a * rho_i
            y0 = b * rho_i

            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

    num_lines = 0 if lines is None else len(lines)
    print(f"[{params['name']}] rho={rho}, threshold={thr}, lines={num_lines}")

    cv2.imshow(f"HoughLines_{params['name']}", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
