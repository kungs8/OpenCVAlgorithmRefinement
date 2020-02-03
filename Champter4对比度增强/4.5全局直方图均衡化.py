# -*- coding: utf-8 -*- 
# @Time : 2020/2/2 19:10 
# @Author : Yanpeng Gong 
# @Site :  
# @File : 4.5全局直方图均衡化.py 
# @Software: PyCharm

# 步骤：
# Step1: 计算图像的灰度直方图
# Step2: 计算灰度直方图的累加直方图
# Step3: 根据累加直方图和直方图均衡化原理得到输入灰度级和输出灰度级之间的映射关系
# Step4: 根据Step3 得到的灰度级映射关系，循环得到输出图像的每一个像素的灰度级。

import cv2
import sys
import numpy as np
import math
# 增加上一层文件的路径
sys.path.append("../")
from General_functions.General_function import calcGrayHist, plotImg


def equalHist(image):
    """
    全局直方图均衡化
    :param image: 需要进行全局直方图均衡化的图像
    :return:
    """
    # 灰度图像矩阵的高、宽
    rows, cols = image.shape
    # Step1：计算灰度直方图
    grayHist = calcGrayHist(image=image)
    # Step2：计算累加直方图
    zeroCumuMoment = np.zeros([256], np.uint32)
    for p in range(256):
        if p == 0:
            zeroCumuMoment[p] = grayHist[0]
        else:
            zeroCumuMoment[p] = zeroCumuMoment[p - 1] + grayHist[p]
    # Step3：根据累加灰度直方图得到输入灰度级和输出灰度级之间的映射关系
    outPut_q = np.zeros([256], np.uint8)
    cofficient = 256.0/(rows * cols)
    for p in range(256):
        q = cofficient * float(zeroCumuMoment[p]) - 1
        if q >= 0:
            outPut_q[p] = math.floor(q)
        else:
            outPut_q[p] = 0
    # Step4：得到直方图均衡化后的图像
    equalHistImage = np.zeros(image.shape, np.uint8)
    for r in range(rows):
        for c in range(cols):
            equalHistImage[r][c] = outPut_q[image[r][c]]
    return equalHistImage


if __name__ == '__main__':
    if len(sys.argv) > 1:
        Img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    else:
        print("Usage: python equalHist.py imageFile")

    # 全局直方图均衡化
    RImg = equalHist(image=Img)

    # 绘出灰度直方图
    plotImg(Img)
    plotImg(RImg)
    # 原图像和均衡化后的图像展示
    cv2.imshow("Img", Img)
    cv2.imshow("RImg", RImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# 运行
# python 4.5全局直方图均衡化.py ../OpenCV_ImgData/Champter4/img1.jpg
