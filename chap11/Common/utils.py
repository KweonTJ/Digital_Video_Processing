import cv2

def put_string(frame, text, pt, value, color=(0,255,0)):
    text = f"{text}{value}"
    cv2.putText(frame, text, pt,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
