# -*- coding: utf-8 -*- 
# @Time : 2020/3/23 14:50 
# @Author : Yanpeng Gong 
# @Site :  
# @File : 5.5双边滤波.py 
# @Software: PyCharm

# ----------------------------------------------------------------------------------------------------------------------
# 图像双边滤波
import math
import sys

import cv2
import numpy as np
def getClosenessWeight(sigma_g, H, W):
    """
    定义函数getClosenessWeight 构建 H * W 的空间距离权重和构建高斯卷积核类似
    :param sigma_g: 空间距离权重模板的标准差
    :param H: 权重模板的高(数值为奇数)
    :param W: 权重模板的宽(数值为奇数)
    :return:
    """
    r, c = np.mgrid[0:H:1, 0:W:1]
    r -= int((H-1) / 2)
    c -= int((W-1) / 2)
    closeWeight = np.exp(-0.5 * (np.power(r, 2) + np.power(c, 2)) / math.pow(sigma_g, 2))
    return closeWeight

def bfltGray(I, H, W, sigma_g, sigma_d):
    """
    图像的双边滤波
    :param I: 图像矩阵,灰度值范围是[0, 1]
    :param H: 权重模板的高(数值为奇数)
    :param W: 权重模板的宽(数值为奇数)
    :param sigma_g: 空间距离权重模板的标准差(sigma_g>1效果会比较好)
    :param sigma_d: 相似性权重模板的标准差(sigma_d<1效果会比较好)
    :return: 浮点型矩阵
    """
    # 构建空间距离权重模板
    closenessWeight = getClosenessWeight(sigma_g, H, W)
    # 模板的中心位置
    cH = (H-1)/2
    cW = (W-1)/2
    # 图像矩阵的行数和列数
    rows, cols = I.shape
    # 双边滤波后的结果
    bfltGrayImage = np.zeros(I.shape, np.float32)
    for r in range(rows):
        for c in range(cols):
            pixel = I[r][c]
            # 判断边界
            rTop = 0 if r - cH < 0 else r - cH
            rBottom = rows - 1 if r + cH > rows - 1 else r + cH
            cLeft = 0 if c - cW < 0 else c - cW
            cRight = cols - 1 if c + cW > cols - 1 else c + cW
            # 权重模板作用的区域
            rTop, rBottom, cLeft, cRight, r, c, cH, cW = int(rTop), int(rBottom), int(cLeft), int(cRight), int(r), int(c), int(cH), int(cW)
            # print(rTop, rBottom, cLeft, cRight)
            region = I[rTop:rBottom+1, cLeft:cRight+1]
            # 构建灰度值相似性的权重因子
            similarityWeightTemp = np.exp(-0.5 * np.power(region - pixel, 2.0) / math.pow(sigma_d, 2))
            closenessWeightTemp = closenessWeight[rTop-r+cH:rBottom-r+cH+1, cLeft-c+cW:cRight-c+cW+1]
            # 两个权重模板相乘
            weightTemp = similarityWeightTemp * closenessWeightTemp
            # 归一化权重模板
            weightTemp = weightTemp / np.sum(weightTemp)
            # 权重模板和对应的邻域值相乘求和
            bfltGrayImage[r][c] = np.sum(region * weightTemp)
    return bfltGrayImage


if __name__ == '__main__':
    if len(sys.argv) > 1:
        image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    else:
        print("Usage: python BFilter.py imageFile")
    # 显示原图
    cv2.imshow("image:", image)
    # 将灰度值归一化
    image = image / 255.0
    # 双边滤波
    bfltImage = bfltGray(I=image, H=33, W=33, sigma_g=19, sigma_d=0.2)
    # 显示双边滤波的结果
    cv2.imshow("BilateralFiltering:", bfltImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# 运行
# python 5.5双边滤波.py ../OpenCV_ImgData/Champter5/img1.png

# 与高斯平滑\均值平滑处理相比较,显然双边滤波在平滑作用的基础上,保持了图像中目标的边缘,
# 但是由于对每个位置都需要重新计算权重模板,所以会非常耗时,一些研究者近几年提出了双边滤波的快速算法.
# ----------------------------------------------------------------------------------------------------------------------