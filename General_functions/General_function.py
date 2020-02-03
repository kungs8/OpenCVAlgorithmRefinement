# -*- coding: utf-8 -*- 
# @Time : 2020/1/29 23:13 
# @Author : Yanpeng Gong 
# @Site :  
# @File : General_function.py 
# @Software: PyCharm

import cv2
import matplotlib.pyplot as plt
import numpy as np


def CVToLocal(ImgName, img):
    """
    保存到本地
    :param ImgName:  保存到本地的图像名
    :param img:  图像数据
    :return:
    """
    cv2.imwrite(filename="../RstData/{}.jpg".format(ImgName), img=img)
    print("{} save over!".format(ImgName))


def plotImg(image):
    """
    绘出直方图
    :param image: 需要绘直方图的图像
    :return:
    """
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


def calcGrayHist(image):
    """
    计算灰度直方图
    :param image: 需要绘直方图的图像
    :return:
    """
    # 灰度图像矩阵的高、宽
    rows, cols = image.shape
    # 存储灰度直方图
    grayHist = np.zeros([256], np.float64)
    for r in range(rows):
        for c in range(cols):
            grayHist[image[r][c]] += 1
    return grayHist