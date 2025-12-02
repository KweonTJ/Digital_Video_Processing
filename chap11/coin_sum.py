from coin_utils import *
from coin_preprocess import *
from Common.utils import put_string

coin_no = int(input("동정 영상 번호 : "))
image, th_img = preprocessing(coin_no)
circles = find_coins(th_img)
coin_imgs = make_coin_img(image, circles)
coin_hists = [calc_histo_hue(coin) for coin in coin_imgs]

groups = grouping(coin_hists)
ncoins = classify_coins(circles, groups)

coin_value = np.array([10, 50, 100, 500])
for i in range(4):
    print("%3d원 : %3d개" %(coin_value[i], ncoins[i]))

total = sum(coin_value * ncoins )
str = "Total coin: {:,} Won".format(total)
print(str)
put_string(image, str, (650, 50), '', (0, 230, 0))

color = [(0,0,255), (255,255,0), (0,250,0), (250, 0, 255)]
for i, (c,r) in enumerate(circles):
    cv2.circle(image, c, r, color[groups[i]], 2)
    put_string(image, i, (c[0] - 15, c[1] - 15), '', color[2])
    put_string(image, i, (c[0], c[1] - 15), '', color[3])

cv2.imshow('result_image', image)
key = cv2.waitKey(0)