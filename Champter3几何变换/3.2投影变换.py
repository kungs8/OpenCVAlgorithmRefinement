# -*- coding: utf-8 -*- 
# @Time : 2020/1/30 16:27 
# @Author : Yanpeng Gong 
# @Site :  
# @File : 3.2投影变换.py 
# @Software: PyCharm
import sys

import cv2
import numpy as np

# cv2.getPerspectiveTransform(src, dst)
# src: 原坐标
# dst: 投影变换后的坐标
# 原理:物体在三维空间中发生了旋转。由于可能出现阴影或者遮挡，所以此投影变换是很难修正的。但如果物体是平面的，那么就能通过二维投影变换对此物体三维变换进行模型化(即：专用的二维投影变换)

# 例：假设(0, 0), (200, 0), (0, 200), (200, 200)是原坐标，通过某投影变换依次转换为(100, 20), (200, 20), (50, 70), (250, 70), 求该投影变换矩阵

src = np.array([[0, 0], [200, 0], [0, 200], [200, 200]], np.float32)
dst = np.array([[100, 20], [200, 20], [50, 70], [250, 70]], np.float32)
P = cv2.getPerspectiveTransform(src=src, dst=dst)
print("P", P)


# P [[ 5.00000000e-01 -3.75000000e-01  1.00000000e+02]
#  [ 3.88578059e-16  7.50000000e-02  2.00000000e+01]
#  [ 9.54097912e-18 -2.50000000e-03  1.00000000e+00]]

# 例：图像在三维空间中进行旋转、平移等变换后的输出
if __name__ == '__main__':
    if len(sys.argv) > 1:
        image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    else:
        print("Usage: python warpPerspect")

    # 原图的高、宽
    h, w = image.shape
    # 原图的四个点与投影变换对应的点
    src = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]], np.float32)
    dst = np.array([[50, 50], [w/3, 50], [50, h-1], [w-1, h-1]], np.float32)
    dst1 = np.array([[50, 50], [w/4, 50], [50, h/4], [w-1, h-1]], np.float32)

    # 计算投影变换矩阵
    p = cv2.getPerspectiveTransform(src=src, dst=dst)
    p1 = cv2.getPerspectiveTransform(src=src, dst=dst1)
    # 利用计算出的投影变换矩阵进行头像的投影变换
    r = cv2.warpPerspective(src=image, M=p, dsize=(w, h), flags=cv2.INTER_LINEAR, borderValue=125)
    r1 = cv2.warpPerspective(src=image, M=p1, dsize=(w, h), flags=cv2.INTER_LINEAR, borderValue=125)

    # 显示原图喝投影效果
    cv2.imshow("image", image)
    cv2.imshow("warpPerspective", r)
    cv2.imshow("warpPerspective1", r1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()