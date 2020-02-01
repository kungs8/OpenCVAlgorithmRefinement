# -*- coding: utf-8 -*- 
# @Time : 2020/2/1 11:34 
# @Author : Yanpeng Gong 
# @Site :  
# @File : 4.1灰度直方图.py 
# @Software: PyCharm
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------------------------------------------------
def calcGrayHist(image):
    # 灰度图像矩阵的高、宽
    rows, cols = image.shape
    # 存储灰度直方图
    grayHist = np.zeros([256], np.float64)
    for r in range(rows):
        for c in range(cols):
            grayHist[image[r][c]] += 1
    return grayHist


if __name__ == '__main__':
    if len(sys.argv) > 1:
        image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    else:
        print("Usage: python histgram.py imageFile")

    # 计算灰度直方图
    grayHist = calcGrayHist(image)
    # 画出灰度直方图
    x_range = range(256)
    plt.plot(x_range, grayHist, 'r', linewidth=2, c="black")
    # 设置坐标轴的范围
    y_maxValue = np.max(grayHist)
    plt.axis([0, 255, 0, y_maxValue])
    # 设置坐标轴的标签
    plt.xlabel("gray Level")
    plt.ylabel("number of pixels")
    # 显示灰度直方图
    plt.show()

# 运行
# python 4.1灰度直方图.py  ../OpenCV_ImgData/Champter4/img1.jpg

# ----------------------------------------------------------------------------------------------------------------------
# matplotlib自身提供的灰度直方图函数hist
if __name__ == '__main__':
    if len(sys.argv) > 1:
        image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    else:
        print("Usage: python histogram.py imageFile")

    # 得到图像矩阵的高、宽
    rows, cols = image.shape
    # 将二维的图像矩阵，变为依偎的数组，便于计算灰度直方图
    pixelSequence = image.reshape([rows * cols, ])
    # 组数
    numberBins = 256
    # 计算灰度直方图
    histogram, bins, patch = plt.hist(pixelSequence, numberBins, facecolor="black", histtype="bar")

    # 设置坐标轴的标签
    plt.xlabel(u"gray Level")
    plt.ylabel(u"number of pixels")
    # 设置坐标轴的范围
    y_maxValue = np.max(histogram)
    plt.axis([0, 255, 0, y_maxValue])
    plt.show()

# 运行
# python 4.1灰度直方图.py  ../OpenCV_ImgData/Champter4/img1.jpg

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
