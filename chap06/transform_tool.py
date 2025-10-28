import sys, os, cv2, numpy as np

WIN = "Click points | r=reset, Enter=run, q/ESC=quit"
pts, mode, img, disp, img_path = [], None, None, None, None

# 한글 경로 안전 로드 함수
def imread_unicode(path):
    path = os.path.normpath(path)
    data = np.fromfile(path, np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def draw_ui():
    global disp
    disp = img.copy()
    h, w = img.shape[:2]
    cv2.rectangle(disp, (0,0), (w,40), (0,0,0), -1)
    msg = f"[{mode.upper()}] clicks:{len(pts)}"
    cv2.putText(disp, msg, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    for i, p in enumerate(pts):
        cv2.circle(disp, p, 5, (0,255,0), -1)
        cv2.putText(disp, f"{i+1}", (p[0]+6, p[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        need = 3 if mode == "affine" else 4 if mode == "persp" else 0
        if len(pts) >= need:  # 이미 다 찍은 경우 무시
            return
        pts.append((x, y))
        draw_ui()
        # 점을 모두 찍으면 자동 변환
        if len(pts) == need:
            compute_and_show()

def compute_and_show():
    global pts, mode
    h, w = img.shape[:2]
    if mode == "affine":
        if len(pts) != 3: print("어파인은 3점 필요"); return
        dst = np.float32([[0,0],[w,0],[w,w]])
        M = cv2.getAffineTransform(np.float32(pts), dst)
        out = cv2.warpAffine(img, M, (w, max(h, w)))
    elif mode == "persp":
        if len(pts) != 4: print("원근은 4점 필요"); return
        dst = np.float32([[0,0],[w,0],[w,h],[0,h]])
        M = cv2.getPerspectiveTransform(np.float32(pts), dst)
        out = cv2.warpPerspective(img, M, (w, h))
    else:
        return

    cv2.imshow("Result", out)
    base, ext = os.path.splitext(img_path)
    out_name = f"{base}_{mode}{ext if ext else '.png'}"
    cv2.imencode(ext if ext else '.png', out)[1].tofile(out_name)
    print(f"저장 완료: {out_name}")

def main():
    global img, disp, mode, pts, img_path
    DEFAULT_PATH = r"C:\Users\user\Desktop\2학기\디지털 영상 처리\chap06\perspective.jpg"
    img_path = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_PATH

    # 1단계: 사용자에게 변환 종류 입력
    print("변환 종류를 선택하세요.")
    print("1: 어파인 변환 (3점)")
    print("2: 원근 변환 (4점)")
    sel = input("선택 (1 또는 2): ").strip()
    if sel == '1': mode = "affine"
    elif sel == '2': mode = "persp"
    else:
        print("잘못된 입력. 종료.")
        return

    # 2단계: 이미지 로드
    img0 = imread_unicode(img_path)
    if img0 is None:
        print(f"이미지 로드 실패: {img_path}")
        return
    img = img0

    # 3단계: 마우스로 점 선택 및 변환 실행
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_TOPMOST, 1)
    draw_ui()

    while True:
        cv2.imshow(WIN, disp)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('r'): pts.clear(); draw_ui()
        elif k in (13, 10): compute_and_show()
        elif k in (ord('q'), 27): break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
