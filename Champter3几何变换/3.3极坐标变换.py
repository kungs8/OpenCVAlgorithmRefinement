# -*- coding: utf-8 -*- 
# @Time : 2020/1/30 18:13 
# @Author : Yanpeng Gong 
# @Site :  
# @File : 3.3极坐标变换.py 
# @Software: PyCharm
import sys

import cv2
import math
import numpy as np

# 例: (11, 13)以(3, 5)为中心进行极坐标变换
r = math.sqrt(math.pow(11-3, 2) + math.pow(13-5, 2))
theta = math.atan2(13-5, 11-3)/math.pi * 180
print("r:{}   theta:{}".format(r, theta))
# r:11.313708498984761   theta:45.0

# -------------------------------------------------------------------
# opencv方法：
# cv2.cartToPolar(x, y[, magnitude[, angle[, angleInDegrees]]])
# x: array 数组且数据类型为浮点型、float32或float64
# y: 和x具有相同尺寸和数据类型的array数组
# angleInDegress: 当值为True, 返回值angle是角度; 反之, 为弧度

# 例：计算(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)这9个点以(1, 1)为中心进行的极坐标变换。
x = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]], np.float64) - 1
y = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], np.float64) - 1
r, theta = cv2.cartToPolar(x, y, angleInDegrees=True)
print("r:\n{}\ntheta:\n{}".format(r, theta))
# r:
# [[1.41421356 1.         1.41421356]
#  [1.         0.         1.        ]
#  [1.41421356 1.         1.41421356]]
# theta:
# [[224.99045634 270.         315.00954366]
#  [180.           0.           0.        ]
#  [135.00954366  90.          44.99045634]]

# -------------------------------------------------------------------
# 极坐标转换为笛卡尔坐标
# cv2.polarToCart(magnitude, angle[, x[, y[, angleInDegrees]]])
# 例：已知极坐标系θ0r中的(30, 10), (31, 10), (30, 11), (31, 11), 其中θ是以角度表示的，问笛卡尔坐标系xoy中的哪四个坐标以(-12, 15)为中心经过极坐标变换后得到这四个坐标
angle = np.array([[30, 31], [30, 31]], np.float32)
r = np.array([[10, 10], [11, 11]], np.float32)
x, y = cv2.polarToCart(r, angle, angleInDegrees=True)
x += -12
y += 15
print("x:\n{}\n y:\n{}".format(x, y))
# x:
# [[-3.3397446 -3.4283257]
#  [-2.4737196 -2.5711575]]
#  y:
# [[20.       20.150383]
#  [20.5      20.66542 ]]

# -------------------------------------------------------------------
# 极坐标变换对图像进行变换
a = np.array([[1, 2], [3, 4]])
b = np.tile(a, (2, 3))  # 将a分别在垂直方向和水平方向上复制2次和3次
print("b:\n", b)
# b:
#  [[1 2 1 2 1 2]
#  [3 4 3 4 3 4]
#  [1 2 1 2 1 2]
#  [3 4 3 4 3 4]]

def polar(image, center, r, theta=(0, 360), rstep=1.0, thetastep=360.0/(180*8)):
    """
    图像的极坐标变换，对于灰度的插值，默认使用的是最近邻插值方法，也可以换成别的插值方法
    :param image: 输入图像
    :param center: 极坐标变换中心
    :param r: 二元元组，代表最小距离和最大距离
    :param theta: 角度范围，默认值是[0, 360]
    :param rstep: r的变换步长
    :param thetastep: 角度的变换步长，默认值是 1/4
    :return:
    """
    # 图像的高、宽
    h, w = image.shape
    # 得到距离最小、最大范围
    minr, maxr = r
    # 角度最小范围
    mintheta, maxtheta = theta
    # 输出图像的高、宽
    H = int((maxr - minr)/rstep) + 1
    W = int((maxtheta - mintheta)/thetastep) + 1
    RImage = 125 * np.ones((H, W), image.dtype) # 这里的125 是颜色的标记，黑色(0), 白色(255)
    # 极坐标变换
    r = np.linspace(minr, maxr, H)
    r = np.tile(r, (W, 1))
    r = np.transpose(r)
    theta = np.linspace(mintheta, maxtheta, W)
    theta = np.tile(theta, (H, 1))
    x, y = cv2.polarToCart(r, theta, angleInDegrees=True)
    # 最近邻插值
    for i in range(H):
        for j in range(W):
            px = int(round(x[i][j])+center[0])
            py = int(round(y[i][j])+center[1])
            if ((px >= 0 and px <= w-1) and (py >= 0 and py <= h-1)):
                RImage[i][j] = image[py][px]
    return RImage

if __name__ == '__main__':
    if len(sys.argv) > 1:
        image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    else:
        print("Usage: python polar.py image")

    # 极坐标变换中心
    cx, cy = 508, 503
    cv2.circle(image, (int(cx), int(cy)), 10, (255.0, 0, 0), 3)
    # 距离的最小、最大半径  # 200 550 270 340
    RImage = polar(image, (cx, cy), r=(200, 550))
    # 旋转
    # cv2.flip(src, dst, flipCode)
    # src: 输入图像矩阵
    # dst: 输出图像矩阵，其尺寸和数据类型与src相同
    # flipCode: >0(src绕 y 轴的镜像处理); =0(src绕 x 轴的镜像处理); <0(src逆时针旋转180°, 即：先绕 x 轴镜像，再绕 y 轴镜像)
    RImage = cv2.flip(src=RImage, flipCode=0)
    # 显示原图和输出图像
    cv2.imshow("image", image)
    cv2.imshow("RImage", RImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 运行
# python 3.3极坐标变换.py.. / OpenCV_ImgData / Champter3 / img2.jpg

# -------------------------------------------------------------------
# 线性极坐标函数(linearPolar)
# cv2.linearPolar(src, dst, center, maxRadius, flags)
# src: 输入图像矩阵(单、多通道矩阵都可以)
# dst: 输出图像矩阵，其尺寸和src是相同的
# center: 极坐标变换中心
# maxRadius: 极坐标变换的最大距离
# flags: 插值算法，同函数resize、warpAffine的插值方法
# 缺点: 1、极坐标变换的步长是不可控制的；2、该函数只能对整个圆内区域，而无法对一个指定的圆环区域进行极坐标变换
# 注: 图像的宽(W)、高(H),角度θ的变换步长thetastep≈360/H, r的变换步长rstep≈maxRadius/W
if __name__ == '__main__':
    if len(sys.argv) > 1:
        src = cv2.imread(sys.argv[1], cv2.IMREAD_ANYCOLOR)
    else:
        print("Usage: python LinearPolar.py image")

    # 图像的极坐标变换
    dst = cv2.linearPolar(src=src, center=(508, 503), maxRadius=550, flags=cv2.INTER_LINEAR)
    # 图像旋转270°
    dst1 = cv2.rotate(src=dst, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)

    # 显示原图
    cv2.imshow("src", src)
    # 显示极坐标变换的结果
    cv2.imshow("dst", dst)
    cv2.imshow("dst1", dst1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -------------------------------------------------------------------
# 对数极坐标函数(logPolar)
# cv2.logPolar(src, dst, center, M, flags)
# src: 输入图像矩阵(单、多通道矩阵都可以)
# dst: 输出图像矩阵，其尺寸和src是相同的
# center: 极坐标变换中心
# M: 系数，该值大一点效果会好一点
# flags: WARP_FILL_OUTLIERS(笛卡尔坐标向对数极坐标变换)、WARP_INVERSE_MAP(对数极坐标向笛卡尔坐标变换)
if __name__ == '__main__':
    if len(sys.argv) > 1:
        src = cv2.imread(sys.argv[1], cv2.IMREAD_ANYCOLOR)
    else:
        print("Usage: python logPolar.py image")

    # 图像的极坐标变换
    M = 100
    dst = cv2.logPolar(src=src, center=(508, 503), M=M, flags=cv2.WARP_FILL_OUTLIERS)
    M1 = 150
    dst1 = cv2.logPolar(src=src, center=(508, 503), M=M1, flags=cv2.WARP_FILL_OUTLIERS)


    # 显示原图
    cv2.imshow("src", src)
    # 显示极坐标变换的结果
    cv2.imshow("dst", dst)
    cv2.imshow("dst1", dst1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 运行
# python 3.3极坐标变换.py  ../OpenCV_ImgData/Champter3/img2.jpg