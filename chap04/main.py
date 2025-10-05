# 명암 대비 코드
import numpy as np, cv2

def imread_u8(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, flags)

IMG_PATH = r"C:\Users\user\Desktop\2학기\디지털 영상 처리\chap04\equalize.jpg"

# ---------- 공용 함수 ----------
def draw_hist_gray(img_gray, bins=256, size=(200,256)):
    hist = cv2.calcHist([img_gray],[0],None,[bins],[0,256]).ravel()
    canvas = np.full(size, 255, np.uint8)
    hist = cv2.normalize(hist, None, 0, size[0], cv2.NORM_MINMAX).ravel()
    gap = size[1] / bins
    for i, v in enumerate(hist):
        x = int(round(i * gap))
        w = max(1, int(round(gap)))
        cv2.rectangle(canvas, (x, size[0]-int(v)), (x+w-1, size[0]-1), 0, cv2.FILLED)
    return canvas

def draw_hist_rgb(img_bgr, bins=256, size=(200,256)):
    colors = [(255,0,0),(0,255,0),(0,0,255)]  # B,G,R
    hist_imgs = []
    for c in range(3):
        hist = cv2.calcHist([img_bgr],[c],None,[bins],[0,256]).ravel()
        canvas = np.full((size[0], size[1], 3), 255, np.uint8)
        hist = cv2.normalize(hist, None, 0, size[0], cv2.NORM_MINMAX).ravel()
        gap = size[1] / bins
        for i, v in enumerate(hist):
            x = int(round(i * gap))
            w = max(1, int(round(gap)))
            cv2.rectangle(canvas, (x, size[0]-int(v)), (x+w-1, size[0]-1), colors[c], cv2.FILLED)
        hist_imgs.append(canvas)
    return hist_imgs

def hist_stretch(img_gray):
    hist = cv2.calcHist([img_gray],[0],None,[256],[0,256]).ravel()
    nz = np.where(hist>0)[0]
    lo, hi = int(nz[0]), int(nz[-1])
    idx = np.arange(256, dtype=np.float32)
    lut = (idx - lo) / (hi - lo) * 255.0
    lut[:lo] = 0
    lut[hi+1:] = 255
    return cv2.LUT(img_gray, lut.astype(np.uint8)), (lo, hi)

def equalize_with_bins(img_gray, bins=256):
    bins = int(max(2, bins))
    hist = cv2.calcHist([img_gray],[0],None,[bins],[0,256]).ravel()
    cdf = np.cumsum(hist)
    cdf = np.round(cdf * 255.0 / (cdf[-1] + 1e-12)).astype(np.uint8)
    edges = np.linspace(0, 256, bins+1, dtype=np.int32)
    lut = np.zeros(256, np.uint8)
    for k in range(bins):
        lut[edges[k]:edges[k+1]] = cdf[k]
    return cv2.LUT(img_gray, lut)

def metrics(img_gray):
    h = cv2.calcHist([img_gray],[0],None,[256],[0,256]).ravel()
    p = h / (h.sum() + 1e-12)
    ent = float(-np.sum(p[p>0]*np.log2(p[p>0])))
    std = float(img_gray.std())
    uniq = int((h>0).sum())
    return ent, std, uniq

# ---------- 1) 컬러 영상 ----------
img_color = imread_u8(IMG_PATH, cv2.IMREAD_COLOR)
if img_color is None: raise Exception("영상 파일 읽기 오류")

cv2.imshow("1. color_image", img_color)
bgr_hists = draw_hist_rgb(img_color)
cv2.imshow("2. B_hist", bgr_hists[0])
cv2.imshow("3. G_hist", bgr_hists[1])
cv2.imshow("4. R_hist", bgr_hists[2])

# ---------- 2) 그레이 영상 ----------
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
hist_gray = draw_hist_gray(img_gray)
cv2.imshow("5. gray_image", img_gray)
cv2.imshow("6. gray_hist", hist_gray)

# ---------- 3) 히스토그램 스트레칭 ----------
stretched, (lo,hi) = hist_stretch(img_gray)
hist_stretch_img = draw_hist_gray(stretched)
cv2.imshow("7. stretching_image", stretched)
cv2.imshow("8. stretching_hist", hist_stretch_img)

# ---------- 4) 히스토그램 평활화 ----------
equalized = cv2.equalizeHist(img_gray)
hist_equalized = draw_hist_gray(equalized)
cv2.imshow("9. equalize_image", equalized)
cv2.imshow("10. equalize_hist", hist_equalized)

# ===== bin-size 효과 분석 (창 추가 없이 콘솔만 출력) =====
bin_list = [256, 128, 64, 32, 16]
print("\n[bin-size 평활화 지표]")
for b in bin_list:
    eqb = equalize_with_bins(img_gray, b)
    ent, std, uniq = metrics(eqb)
    print(f"bins={b:>3} | entropy={ent:.3f} | std={std:.2f} | unique_levels={uniq}")

cv2.waitKey(0)
cv2.destroyAllWindows()
