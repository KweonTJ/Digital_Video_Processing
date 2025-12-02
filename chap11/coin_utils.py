import numpy as np, cv2

def make_coin_img(src, circles):
    coins = []
    for center, radius in circles:
        r=radius * 3
        cen = (r // 2, r // 2)
        mask = np.zeros((r,r,3), np.uint8)
        cv2.circle(mask, cen, radius, (255,255,255), cv2.FILLED)

        coin = cv2.getRectSubPix(src, (r,r), center)
        coin = cv2.bitwise_and(coin, mask)
        coins.append(coin)
    return coins

def calc_histo_hue(coins):
    hsv = cv2.cvtColor(coins, cv2.COLOR_BGR2HSV)
    hsize, ranges = [32], [0, 180]
    hist = cv2.calcHist([hsv], [0], None, hsize, ranges)
    return hist.flatten()

def classify_coins(circles, groups):
    ncoins = [0] * 4
    g = np.full((2,70), -1, np.int32) # numpy 버전 이슈로 np.int32를 활용하였습니댜.
    g[0, 26:47], g[0, 47:50], g[0, 50:] = 0, 2, 3
    g[1, 36:44], g[1, 44:50], g[1, 50:] = 1, 2, 3

    for group, (_, radius) in zip(groups, circles):
        coin = g[group, radius]
        ncoins[coin] += 1

    return np.array(ncoins)

def grouping(hists):
    ws = [0,0,0,0,0,0,0,0,
          0,0,0,0,0,1,2,3,
          4,5,6,8,6,5,4,3,
          2,1,0,0,0,0,0,0]

    sim = np.multiply(hists, ws)
    similaritys = np.sum(sim, axis=1) / np.sum(hists, axis=1)

    groups = [1 if s > 1.2 else 0 for s in similaritys]
    return groups
