# -*- encoding: utf-8 -*-
'''
@File    :   12.3RGB.py
@Time    :   2022/03/27 18:35:27
@Author  :   yanpenggong
@Version :   1.0
@Email   :   yanpenggong@163.com
@Copyright : 侵权必究
'''

# here put the import lib
# ----------------------------------------------------------------------------------------------------------------------
import sys
import cv2

def main():
    if len(sys.argv) > 1:
        image = cv2.imread(filename=sys.argv[1], flags=cv2.IMREAD_COLOR)
    else:
        print("Usage: python **.py imageFile")
    # 得到三个颜色通道
    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]

    # 8位图转换为 浮点型
    fImg = image / 255.0
    fb = fImg[:, :, 0]
    fg = fImg[:, :, 1]
    fr = fImg[:, :, 2]

    # 显示 原图 及 三个颜色
    cv2.imshow(winname="image", mat=image)
    cv2.imshow(winname="b", mat=b)
    cv2.imshow(winname="g", mat=g)
    cv2.imshow(winname="r", mat=r)
    cv2.imshow(winname="fb", mat=fb)
    cv2.imshow(winname="fg", mat=fg)
    cv2.imshow(winname="fr", mat=fr)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    