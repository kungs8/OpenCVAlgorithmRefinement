# -*- coding: utf-8 -*- 
# @Time : 2020/3/2 23:38 
# @Author : Yanpeng Gong 
# @Site :  
# @File : 5.2高斯平滑.py 
# @Software: PyCharm

# ----------------------------------------------------------------------------------------------------------------------
# 5.2.1 高斯卷积核的构建及分离性
# 构建高斯卷积算子
import math

import numpy as np

def getGaussKernel(sigma, H, W):
    # 第一步：构建高斯矩阵
    gaussMatrix = np.zeros([H, W], np.float32)
    # 得到中心点的位置
    cH = (H - 1) / 2
    cW = (W - 1) / 2
    # 计算gauss(sigma, r, c)
    for r in range(H):
        for c in range(W):
            norm2 = math.pow(r - cH, 2) + math.pow(c - cW, 2)
            gaussMatrix[r][c] = math.exp(-norm2 / (2 * math.pow(sigma, 2)))
    # s第二步：计算高斯矩阵的和
    sumGM = np.sum(gaussMatrix)
    # 第三步：归一化
    gaussKernel = gaussMatrix / sumGM
    return gaussKernel

sigma=0.5
H = 3
W = 3
res = getGaussKernel(sigma, H, W)
print("res:\n", res)
# res:
#  [[0.01134374 0.0838195  0.01134374]
#  [0.0838195  0.619347   0.0838195 ]
#  [0.01134374 0.0838195  0.01134374]]

# 注: 因为最后要归一化，所以再代码实现中可以去掉高斯函数中的系数( 1/(2*pi*sigma^2) )
# 上面的代码第一步"计算高斯矩阵"，也可以用Numpy中的函数power和exp进行简化
r, c = np.mgrid[0:H:1, 0:W:1]
r = r - (H-1)/2
c = c - (W-1)/2
gaussMatrix = np.exp(-0.5 * (np.power(r, 2) + np.power(c, 2)) /math.pow(sigma, 2))
print("gaussMatrix:\n", gaussMatrix)
# gaussMatrix:
#  [[0.01831564 0.13533528 0.01831564]
#  [0.13533528 1.         0.13533528]
#  [0.01831564 0.13533528 0.01831564]]

# ----------------------------------------------------------------------------------------------------------------------
# 高斯卷积算子是可分离卷积核
# cv2.getGaussianKernel(ksize, sigma, ktype)
# ksize: 一维垂直方向上高斯核的行数，而且是正奇数
# sigma: 标准差
# ktype: 返回值的数据类型为CV_32F或CV_64F, 默认是CV_64F
# 注: 返回值是一个ksize * 1 的垂直方向上的高斯核，而对于水平方向上的高斯核，只需对垂直方向上的高斯核进行转置就可以了

import cv2
import numpy as np
gk = cv2.getGaussianKernel(3, 2, cv2.CV_64F)
print("gk:\n", gk)
# gk:
#  [[0.31916777]
#  [0.36166446]
#  [0.31916777]]

# ----------------------------------------------------------------------------------------------------------------------
# 5.2.3 定义函数gaussBlur实现图像的高斯平滑，后面两个参数同 signal.convolve2d的，用来表示图像的边界扩充方式，常用方式：'symm'反射扩充

from scipy import signal
import sys

def gaussBlur(image, sigma, H, W, _boundary='fill', _fillvalue=0):
    """
    图像的高斯平滑
    :param image: 需要进行高斯平滑的图像
    :param sigma: 高斯卷积核的标准差
    :param H: 高斯卷积核的高
    :param W: 高斯卷积核的宽
    :param _boundary: 边界填充方式
    :param _fillvalue: 边界填充的数值
    :return:
    """
    # 构建水平方向上的高斯卷积核
    gaussKernel_x = cv2.getGaussianKernel(ksize=W, sigma=sigma, ktype=cv2.CV_64F)
    # 转置
    gaussKernel_x = np.transpose(gaussKernel_x)
    # 图像矩阵与水平高斯核卷积
    gaussBlur_x = signal.convolve2d(image, gaussKernel_x, mode='same', boundary=_boundary, fillvalue=_fillvalue)
    # 构建垂直方向上的高斯卷积核
    gaussKernel_y = cv2.getGaussianKernel(ksize=H, sigma=sigma, ktype=cv2.CV_64F)
    # 与垂直方向上的高斯核卷积核
    gaussBlur_xy = signal.convolve2d(gaussBlur_x, gaussKernel_y, mode='same', boundary=_boundary, fillvalue=_fillvalue)
    return gaussBlur_xy


if __name__ == '__main__':
    if len(sys.argv) > 1:
        image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    else:
        print("Usage: python gaussBlur.py imageFile")
    cv2.imshow("image:", image)
    #  高斯平滑
    blurImage = gaussBlur(image, sigma=5, H=35, W=35, _boundary="symm")
    # 对blurImage进行灰度级显示
    blurImage = np.round(blurImage)
    blurImage = blurImage.astype(np.uint8)
    cv2.imshow("GaussBlur:", blurImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# 运行：
# python 5.2高斯平滑.py ../OpenCV_ImgData/Champter5/img3.jpg
# ----------------------------------------------------------------------------------------------------------------------