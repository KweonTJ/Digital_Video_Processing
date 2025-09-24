import numpy as np
import cv2

def onMouse(event, x, y, flags, param):
    global title, pt

    if event == cv2.EVENT_LBUTTONDOWN:
        if pt[0] < 0: pt = (x,y)
        else:
            # 첫 클릭(pt)은 중심, 두번째 클릭(x,y)은 꼭짓점으로 사각형 생성
            dx, dy = abs(pt[0] - x), abs(pt[1] - y)
            cv2.rectangle(image, (pt[0] - dx, pt[1] - dy), (pt[0] + dx, pt[1] + dy), (255,0,0), 2)
            cv2.imshow(title, image)
            pt = (-1,-1)

    elif event == cv2.EVENT_RBUTTONUP:
        if pt[0] < 0: pt = (x,y)
        else :
            # 첫 클릭(pt)과 두번째 클릭(x,y)을 지름의 양 끝점으로 원 생성
            center_x, center_y = (pt[0] + x) // 2, (pt[1] + y) // 2 # 두 점의 중점을 원의 중심으로 계산
            dx, dy = pt[0] - x, pt[1] - y
            radius = int(np.sqrt(dx*dx + dy*dy) / 2) # 두 점 사이 거리의 절반을 반지름으로 계산
            cv2.circle(image, (center_x, center_y), radius, (0,0,255), 2)
            cv2.imshow(title, image)
            pt = (-1,-1)

image = np.full((300, 500, 3), (255,255,255), np.uint8)

pt = (-1,-1)
title = "Draw Event"
cv2.imshow(title, image)
cv2.setMouseCallback(title, onMouse)
cv2.waitKey(0)