# -*- coding: utf-8 -*- 
# @Time : 2020/2/2 14:06 
# @Author : Yanpeng Gong 
# @Site :  
# @File : 4.4伽马变换.py 
# @Software: PyCharm

import cv2
import sys
import numpy as np
# 增加上一层文件路径
sys.path.append("../")
from General_functions.General_function import plotImg

# ----------------------------------------------------------------------------------------------------------------------
# 图像的伽马变换实质是对图像矩阵的每一个值进行幂运算

I = np.array([[1, 2], [3, 4]])
O = np.power(I, 2)  # 对 I 中的每一个值求平方
print("O:", O)
# O: [[ 1  4]
#  [ 9 16]]

# ----------------------------------------------------------------------------------------------------------------------
# 图像的伽马变换，首先将图像的灰度值归一化到[0, 1]范围
if __name__ == '__main__':
    # 读取图像
    if len(sys.argv) > 1:
        I = cv2.imread(sys.argv[1], flags=cv2.IMREAD_GRAYSCALE)
    else:
        print("Usage: python gammaImg.py imageFile")

    # 图像归一化
    fI = I/255.0
    # 伽马变换
    gamma = 0.5
    RImg = np.power(fI, gamma)

    # 显示灰度直方图
    plotImg(image=I)
    plotImg(image=RImg)
    # 显示原图和伽马变换后的效果
    cv2.imshow("I", I)
    cv2.imshow("RImg", RImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# 运行
# python 4.4伽马变换.py   ../OpenCV_ImgData/Champter4/img8.jpg
