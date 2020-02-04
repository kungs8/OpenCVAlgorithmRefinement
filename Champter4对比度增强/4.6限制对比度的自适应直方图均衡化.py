# -*- coding: utf-8 -*- 
# @Time : 2020/2/3 10:29 
# @Author : Yanpeng Gong 
# @Site :  
# @File : 4.6限制对比度的自适应直方图均衡化.py 
# @Software: PyCharm

# 自适应直方图均衡化首先将图像划分为不重叠的区域块(tiles)，然后对每一个块分别进行直方图均衡化。
# 显然，在没有噪声影响的情况下，每一个小区域的灰度直方图会被限制在一个小的灰度级范围内；但是如果有噪声，每一个分割的区域块执行直方图均衡化后，噪声会被放大。
# 为了避免出现噪声这种情况，提出了"限制对比度"(Contrast Limiting)，如果直方图的bin超过了提前预设好的"限制对比度"，那么会被裁减，然后将裁减的部分均匀分布到其它的bin，这样就重构了直方图。
# 例：设置"限制对比度"=40， 共有5个bin的值，第4个bin的值=45(大于40)， 然后将多余的45-40=5均匀分布到每一个bin。

import cv2
import sys
sys.path.append("../")
from General_functions.General_function import equalHist

if __name__ == '__main__':
    if len(sys.argv) > 1:
        src = cv2.imread(sys.argv[1], cv2.IMREAD_ANYCOLOR)
    else:
        print("Usage: python CLACHE.py imageFile")

    # 全局直方图均衡化(目的：给本节形成对比)
    allgray = equalHist(src)

    # 创建CLAHE对象
    clache = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # clipLimit(颜色对比度的阈值)  tileGridSize(进行像素均衡化的网格大小，即在多少网格下进行直方图的均衡化操作)
    # 限制对比度的自适应阈值均衡化
    dst = clache.apply(src)

    # 显示原始图像和限制对比度的自适应直方图均衡化的图
    cv2.imshow("src", src)
    cv2.imshow("allgray", allgray)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 运行
# python 4.6限制对比度的自适应直方图均衡化.py ../OpenCV_ImgData/Champter4/img3.jpg