# -*- coding: utf-8 -*- 
# @Time : 2020/3/16 17:13 
# @Author : Yanpeng Gong 
# @Site :  
# @File : 5.3均值平滑.py 
# @Software: PyCharm

# ----------------------------------------------------------------------------------------------------------------------
# 图像的积分实现:
# 对图像矩阵进行按行积分，然后再按列积分(或者先列积分后行积分)
# 为了再快速均值平滑中省去判断边界的问题，对积分后图像矩阵的上边和左边进行补零操作，尺寸为(R + 1) * (C + 1)
import sys

import cv2
import numpy as np


def integral(image):
    """
    图像的积分
    :param image: 原始图像
    :return:
    """
    rows, cols = image.shape
    # 行积分运算
    inteImageC = np.zeros((rows, cols), np.float32)
    for r in range(rows):
        for c in range(cols):
            if c == 0:
                inteImageC[r][c] = image[r][c]
            else:
                inteImageC[r][c] = inteImageC[r][c-1]+image[r][c]
    # 列积分运算
    inteImage = np.zeros(image.shape, np.float32)
    for c in range(cols):
        for r in range(rows):
            if r == 0:
                inteImage[r][c] = inteImageC[r][c]
            else:
                inteImage[r][c] = inteImage[r-1][c] + inteImageC[r][c]
    # 上边和左边进行补零
    inteImage_0 = np.zeros((rows+1, cols+1), np.float32)
    inteImage_0[1:rows+1, 1:cols+1] = inteImage
    return inteImage_0

# 实现了图像积分后，通过定义函数 fastMeanBlur 来实现均值平滑。
# 如果在图像的边界进行处理是补零操作，那么随着窗口的增大，平滑后黑色边界会越来越明显，所以在进行均值平滑处理时，比较理想的边界扩充类型是镜像扩充。
def fastMeanBlur(image, winSize, borderType=cv2.BORDER_DEFAULT):
    """
    图像均值平滑
    :param image: 输入矩阵
    :param winSize: 平滑窗口尺寸(宽、高均为奇数)
    :param borderType: 边界扩充类型
    :return: 返回浮点型(如果输入的是8位图，则需要利用命令astype(numpy.unit8)将结果转为8位图)
    """
    halfH = (winSize[0]-1)/2
    halfW = (winSize[1]-1)/2
    halfH, halfW = int(halfH), int(halfW)
    ratio = 1.0/(winSize[0]*winSize[1])
    print("halfH:{},halfW:{}".format(halfH, halfW))
    # 边界扩充
    paddImage = cv2.copyMakeBorder(src=image, top=halfH, bottom=halfH, left=halfW, right=halfW, borderType=borderType)
    # 图像积分
    paddIntegral = integral(paddImage)
    # 图像的高、宽
    rows, cols = image.shape
    # 均值滤波后的结果
    meanBlurImage = np.zeros(image.shape, np.float32)
    r, c = 0, 0
    for h in range(halfH, halfH+rows, 1):
        for w in range(halfW, halfW+cols, 1):
            meanBlurImage[r][c] = (paddIntegral[h+halfH+1][w+halfW+1] + paddIntegral[h-halfH][w-halfW] - paddIntegral[h+halfH+1][w-halfW] - paddIntegral[h-halfH][w+halfH+1])*ratio
            c += 1
        r += 1
        c = 0
    return meanBlurImage


if __name__ == '__main__':
    if len(sys.argv) > 1:
        image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    else:
        print("Usage: python gaussBlur.py imageFile")
    cv2.imshow("image:", image)
    # 均值平滑
    meanBlurImage = fastMeanBlur(image=image, winSize=(7, 7))
    cv2.imshow("GaussBlur:", meanBlurImage)
    # 对blurImage进行灰度级显示
    blurImage = np.round(meanBlurImage)
    blurImage = blurImage.astype(np.uint8)
    cv2.imshow("GaussBlur:", blurImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# 结果：
# 均值平滑算子对图的平滑，显然随着均值平滑算子窗口的增大，处理后细节部分越来越不明显，只是显示了大概轮廓.

# 运行：
# python 5.3均值平滑.py ../OpenCV_ImgData/Champter5/img2.png
# ----------------------------------------------------------------------------------------------------------------------