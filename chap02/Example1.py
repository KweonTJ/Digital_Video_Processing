import numpy as np
import cv2

add_value = 10
# 기존 코드의 경우, onChange 안에 있는 변수라서 지역 변수로 추가하였습니다.

def onChange(value):
    global image, title

    add_value = value - int(image[0][0])
    image[:]=image+add_value
    cv2.imshow(title, image)

def onMouse(event, x, y, flags, param):
    global image, bar_name

    if event == cv2.EVENT_RBUTTONDOWN:
        if (image[0][0] <  246):
            image[:] = image + add_value
        cv2.setTrackbarPos(bar_name, title, image[0][0])
        cv2.imshow(title, image)

    elif event == cv2.EVENT_LBUTTONDOWN:
        if (image[0][0] >= 10):
            image[:]=image-10
        cv2.setTrackbarPos(bar_name, title, image[0][0])
        cv2.imshow(title, image)

image = np.zeros((300, 500), np.uint8)
title = "trackbar & Mous Envet"
bar_name = 'Brightness'
cv2.imshow(title, image)

cv2.createTrackbar(bar_name, title, image[0][0], 255, onChange)
cv2.setMouseCallback(title, onMouse)
cv2.waitKey(0)
cv2.destroyAllWindows()
