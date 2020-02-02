# -*- coding: utf-8 -*- 
# @Time : 2020/2/2 10:59 
# @Author : Yanpeng Gong 
# @Site :  
# @File : 4.3直方图正规化.py 
# @Software: PyCharm

import sys
import cv2
import numpy as np
sys.path.append("../")
from General_functions.General_function import plotImg


# ----------------------------------------------------------------------------------------------------------------------
# 直方图正规化
# a = (Omax - Omin)/(Imax - Imin)
# b = Omin - a * Imin
# O = a * I + b
if __name__ == '__main__':
    if len(sys.argv) > 1:
        # 读取图像
        I = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    else:
        print("Usage: python histgray.py image")
    # 求I的最大值、最小值
    Imax = np.max(I)
    Imin = np.min(I)
    # 要输出的最小灰度级和最大灰度级
    Rmin, Rmax = 0, 255
    # 计算 a和b的值
    a = float(Rmax - Rmin)/(Imax - Imin)
    b = Rmin - a * Imin
    # 矩阵的线性变换
    RImg = a * I + b
    # 数据类型转换
    RImg = RImg.astype(np.uint8)

    # 绘出直方图
    plotImg(I)
    plotImg(RImg)
    # 显示原图和直方图正规化的效果
    cv2.imshow("img", I)
    cv2.imshow("RImg", RImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# 运行
# python 4.2线性变换.py  ../OpenCV_ImgData/Champter4/img4.jpg

# ----------------------------------------------------------------------------------------------------------------------
# 正规化函数(normalize)
# cv2.normalize(src, dst, alpha, beta, norm_type, dtype)
# src: 输入矩阵
# dst: 结构元
# alpha: 结构元的锚点
# beta: 腐蚀操作的次数
# norm_type: 边界扩充类型
    # 例: src = [[-55, 80], [100, 255]]
    # 1、NORM_L1(计算矩阵中的绝对值的和)
        # src_type = |-55|+|80|+|100|+|255| = 490
    # 2、NORM_L2(计算矩阵中的平方和的开平方)
        # src_type = np.sqrt(|-55|^2+|80|^2+|100|^2+|255|^2) = 290.6
    # 3、NORM_INF(计算矩阵中的绝对值的最大值)
        # src_type = max(|-55|+|80|+|100|+|255|) = 255
    # dst = alpha * (src/src_type)+beta
    # 4、NORM_MINMAX()
        # src_min = -55, src_max = 255
        # dst = alpha * ((src(r,c) - src_min)/(src_max - src_min)) + beta
# dtype: 边界扩充值
    # CV_8U - 8位无符号整数（0..255）
    # CV_8S - 8位有符号整数（-128..127）
    # CV_16U - 16位无符号整数（0..65535）
    # CV_16S - 16位有符号整数（-32768..32767）
    # CV_32S - 32位有符号整数（-2147483648..2147483647）
    # CV_32F - 32位浮点数（-FLT_MAX..FLT_MAX，INF，NAN）
    # CV_64F - 64位浮点数（-DBL_MAX..DBL_MAX，INF，NAN）

if __name__ == '__main__':
    if len(sys.argv) > 1:
        src = cv2.imread(sys.argv[1], flags=cv2.IMREAD_ANYCOLOR)
    else:
        print("Usage: python normlize.py imageFile")
    # 直方图正规化
    dst = cv2.normalize(src=src, dst=(0, 255), alpha=255, beta=0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 显示原图和直方图正规化效果
    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()