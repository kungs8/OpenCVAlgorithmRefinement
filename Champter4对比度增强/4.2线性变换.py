# -*- coding: utf-8 -*- 
# @Time : 2020/2/1 17:27 
# @Author : Yanpeng Gong 
# @Site :  
# @File : 4.2线性变换.py 
# @Software: PyCharm
import sys

import cv2
import numpy as np
# 增加上一路径到sys中，避免取文件夹中model报错
sys.path.append('../')
from General_functions.General_function import plotImg


# ----------------------------------------------------------------------------------------------------------------------
# 普通的ndarray 进行线性变换
I= np.array([[0, 200], [23, 4]], np.uint8)
res = 2 * I
print("res:\n", res)
print("res_dtype:", res.dtype)
# res:
#  [[  0 144]
#  [ 46   8]]
# res_dtype: uint8

res1 = 2.0 * I
print("res1:\n", res1)
print("res1_dtype:", res1.dtype)
# res1:
#  [[  0. 400.]
#  [ 46.   8.]]
# res1_dtype: float64

# 注：上面的代码中，输入的是一个uint8类型的ndarray， 用数字2乘以该数组，返回的ndarray的数据类型是uint8。
# uint8 范围是[0, 255]，而上面的2 * 200 = 400 超出范围，即：400 % 256 = 144 转为uint8类型
# float64类型 * uint8类型，返回的类型是float64， 这里 2 * 200 = 400 对8位图进行对比增强来说，要截断计算大于255的值

# ----------------------------------------------------------------------------------------------------------------------
# 图像矩阵线性变换

# def plotImg(image):
#     # 得到图像矩阵的高、宽
#     rows, cols = image.shape
#     # 将二维的图像矩阵，变为依偎的数组，便于计算灰度直方图
#     pixelSequence = image.reshape([rows * cols, ])
#     # 组数
#     numberBins = 256
#     # 计算灰度直方图
#     histogram, bins, patch = plt.hist(pixelSequence, numberBins, facecolor="black", histtype="bar")
#
#     # 设置坐标轴的标签
#     plt.xlabel(u"gray Level")
#     plt.ylabel(u"number of pixels")
#     # 设置坐标轴的范围
#     y_maxValue = np.max(histogram)
#     plt.axis([0, 255, 0, y_maxValue])
#     plt.show()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # 读取图像
        I = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    else:
        print("Usage: python lineContrast.py imageFile")

    # 线性变换
    a = 2
    RImg = float(a) * I
    # 进行数据截断，大于255的值要截断为255
    RImg[RImg>255] = 255
    # 数据类型转换
    RImg = np.round(RImg)  # 取整，若四舍五入时到两边的距离一样，结果是偶数部分的值(np.round(0.5)=0.0, np.round(1.5)=2.0)
    RImg = RImg.astype(np.uint8)

    # 绘出直方图
    plotImg(I)
    plotImg(RImg)
    # 显示原图和线性变换后的效果
    cv2.imshow("img", I)
    cv2.imshow("RImg", RImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 运行
# python 4.2线性变换.py  ../OpenCV_ImgData/Champter4/img4.jpg
