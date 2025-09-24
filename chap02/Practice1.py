import cv2
import numpy as np

# 아무 작업도 하지 않는 더미 함수
def nothing(x):
    pass

# 마우스 콜백 함수
def mouse_callback(event, x, y, flags, param):
    global img
    # 왼쪽 버튼 클릭 시
    if event == cv2.EVENT_LBUTTONDOWN:
        img[:] = (255, 255, 255)  # 이미지를 흰색으로 변경
        cv2.setTrackbarPos('R', 'image', 255)
        cv2.setTrackbarPos('G', 'image', 255)
        cv2.setTrackbarPos('B', 'image', 255)

    # 오른쪽 버튼 클릭 시
    elif event == cv2.EVENT_RBUTTONDOWN:
        img[:] = (0, 0, 0)  # 이미지를 검은색으로 변경
        cv2.setTrackbarPos('R', 'image', 0)
        cv2.setTrackbarPos('G', 'image', 0)
        cv2.setTrackbarPos('B', 'image', 0)

# 이미지 및 창 생성
width, height = 512, 512
img = np.zeros((height, width, 3), dtype=np.uint8)
namedWindow = 'image'
cv2.namedWindow(namedWindow)

# 마우스 콜백 함수 설정
cv2.setMouseCallback(namedWindow, mouse_callback)

# 트랙바 생성
cv2.createTrackbar('R', namedWindow, 0, 255, nothing)
cv2.createTrackbar('G', namedWindow, 0, 255, nothing)
cv2.createTrackbar('B', namedWindow, 0, 255, nothing)

while True:
    # 현재 트랙바의 위치를 가져옴
    r = cv2.getTrackbarPos('R', namedWindow)
    g = cv2.getTrackbarPos('G', namedWindow)
    b = cv2.getTrackbarPos('B', namedWindow)

    if cv2.waitKey(1) == -1: # 키 입력이 없을 때
        img[:] = [b, g, r]

    # 이미지 표시
    cv2.imshow(namedWindow, img)

    # 키보드 입력 대기
    key = cv2.waitKey(1) & 0xFF

    # 'ESC' 키를 누르면 루프 종료
    if key == 27:
        break
    # 'R' 키를 누르면 빨간색으로 변경
    elif key == ord('r'):
        img[:] = (0, 0, 255)
        cv2.setTrackbarPos('R', namedWindow, 255)
        cv2.setTrackbarPos('G', namedWindow, 0)
        cv2.setTrackbarPos('B', namedWindow, 0)
    # 'G' 키를 누르면 초록색으로 변경
    elif key == ord('g'):
        img[:] = (0, 255, 0)
        cv2.setTrackbarPos('R', namedWindow, 0)
        cv2.setTrackbarPos('G', namedWindow, 255)
        cv2.setTrackbarPos('B', namedWindow, 0)
    # 'B' 키를 누르면 파란색으로 변경
    elif key == ord('b'):
        img[:] = (255, 0, 0)
        cv2.setTrackbarPos('R', namedWindow, 0)
        cv2.setTrackbarPos('G', namedWindow, 0)
        cv2.setTrackbarPos('B', namedWindow, 255)

# 모든 창 닫기
cv2.destroyAllWindows()