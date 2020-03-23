
# -*- coding: utf-8 -*-
# @Time : 2020/3/17 11:51 
# @Author : Yanpeng Gong 
# @Site :  
# @File : 5.4中值平滑.py 
# @Software: PyCharm

# ----------------------------------------------------------------------------------------------------------------------\
# 5.4.1 原理
# 中值滤波最重要的能力是去除椒盐噪声.
# 椒盐噪声是指在图像传输系统中由于解码误差等原因,导致图像中出现孤立的白点或者黑点.
# 例: 对图像模拟添加椒盐噪声
import numpy as np
import random
import sys
import cv2

def salt(image, number):
    """
    给图像模拟添加椒盐噪声
    :param image: 原始图像
    :param number: 椒盐噪声的数量
    :return: 添加椒盐噪声后的图像
    """
    # 图像的高/宽
    rows, cols = image.shape
    # 加入椒盐噪声后的图像
    saltImage = np.copy(image)
    for i in range(number):
        randR = random.randint(0, rows-1)
        randC = random.randint(0, cols-1)
        saltImage[randR][randC] = 255
    return saltImage


if __name__ == '__main__':
    if len(sys.argv) > 1:
        image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    else:
        print("Usage: python salt.py imageFile")
    # 原图显示
    cv2.imshow("image:", image)
    saltImage = salt(image=image, number=2000)
    saltImage = np.round(saltImage)
    saltImage = saltImage.astype(np.uint8)
    cv2.imshow("saltImage", saltImage)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 运行:
# python 5.4中值平滑.py ../OpenCV_ImgData/Champter5/img2.png

# ----------------------------------------------------------------------------------------------------------------------
# 5.4.2 中值平滑实现
# 首先 ndarray[r1:r2+1, c1:c2+1]得到ndarray从左上角(r1, c1)至右下角(r2, c2)的矩形区域,然后利用 Numpy提供的函数median取该区域的中数
def medianBlur(image, winSize):
    """
    图像的中值平滑
    :param image: 需要中值平滑的图像
    :param winSize: 平滑窗口的高/宽
    :return:
    """
    # 图像的高/宽
    rows, cols = image.shape
    # 窗口的高/宽,均为奇数
    winH, winW = winSize[0], winSize[1]
    halfWinH = (winH-1)/2
    halfWinW = (winW-1)/2
    # 中值滤波后的输出图像
    medianBlurImage = np.zeros(image.shape, image.dtype)
    for r in range(rows):
        for c in range(cols):
            # 判断边界
            rTop = 0 if r - halfWinH < 0 else r - halfWinH
            rBottom = rows - 1 if r + halfWinH > rows - 1 else r + halfWinH
            cLeft = 0 if c - halfWinW < 0 else c - halfWinW
            cRight = cols - 1 if c + halfWinW > cols - 1 else c + halfWinW
            rTop, rBottom, cLeft, cRight = int(rTop), int(rBottom), int(cLeft), int(cRight)
            # 取邻域
            region = image[rTop:rBottom+1, cLeft:cRight+1]
            # 求中值
            medianBlurImage[r][c] = np.median(region)
    return medianBlurImage


if __name__ == '__main__':
    if len(sys.argv) > 1:
        image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    else:
        print("Usage: python medianBlur.py imageFile")
    # 显示原图
    cv2.imshow("image", image)
    # 中值滤波
    medianBlurImage = medianBlur(image=image, winSize=(5, 5))
    # 显示中值滤波后的结果
    cv2.imshow("medianBlurImage:", medianBlurImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 运行5
# python 5.4中值平滑.py ../OpenCV_ImgData/Champter5/img6.jpg
# ----------------------------------------------------------------------------------------------------------------------