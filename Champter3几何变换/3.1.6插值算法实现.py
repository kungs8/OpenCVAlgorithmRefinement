# -*- coding: utf-8 -*- 
# @Time : 2020/1/29 22:36 
# @Author : Yanpeng Gong 
# @Site :  
# @File : 3.1.6图像几何变换.py 
# @Software: PyCharm


import cv2
import sys
import numpy as np


# cv2.warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]])
# src: 输入图像矩阵
# M: 2行3列的仿射变换矩阵
# dsize: 二元元组(宽、高), 输出图像的大小
# flags(插值法): INTE_NEAREST、INTE_LINEAR(默认)等
# borderMode(填充模式): BORDER_CONSTANT等
# borderValue: 当borderMode=BORDER_CONSTANT时的填充值
sys.path.append('../')
print(sys.path)
from General_functions.General_function import CVToLocal

if __name__ == '__main__':
    if len(sys.argv) > 1:
        image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    else:
        print("Usage: python warpAffine.py image")

    # 保存到本地
    CVToLocal(ImgName="warpAffine", img=image)

    # 原图的高、宽
    h, w = image.shape[:2]

    # 仿射变换矩阵，缩小2倍
    A1 = np.array([[0.5, 0, 0], [0, 0.5, 0]], np.float32)
    # d1 = cv2.warpAffine(src=image, M=A1, dsize=(w, h), flags=cv2.INTER_LINEAR, borderValue=125)
    # 插值法flags换成cv2.INTER_NEAREST
    d1 = cv2.warpAffine(src=image, M=A1, dsize=(w, h), flags=cv2.INTER_NEAREST, borderValue=125)

    # 先缩小2倍，再平移(w/4, h/5)
    A2 = np.array([[0.5, 0, w/4], [0, 0.5, h/4]], np.float32)
    d2 = cv2.warpAffine(src=image, M=A2, dsize=(w, h), borderValue=125)

    # 在d2的基础上，绕图像的中心点旋转30°
    A3 = cv2.getRotationMatrix2D(center=(w/2.0, h/2.0), angle=30, scale=1)
    d3 = cv2.warpAffine(src=d2, M=A3, dsize=(w, h), borderValue=125)

    # 图像的展示
    cv2.imshow("img", image)
    cv2.imshow("d1", d1)
    cv2.imshow("d2", d2)
    cv2.imshow("d3", d3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 运行
# ipython 3.1.6插值算法实现.py ../OpenCV_ImgData/Champter3/img.jpg
