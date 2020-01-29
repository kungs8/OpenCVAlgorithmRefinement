# -*- coding: utf-8 -*- 
# @Time : 2020/1/29 21:25 
# @Author : Yanpeng Gong 
# @Site :  
# @File : 3.1.4计算仿射矩阵(1)_方程法.py 
# @Software: PyCharm


import numpy as np
import cv2
# 例：对空间坐标先等比例缩放2倍，然后在水平方向上平移100， 在垂直方向上平移200，计算仿射变换矩阵代码如下

s = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])  # 缩放矩阵
t = np.array([[1, 0, 100], [0, 1, 200], [0, 0, 1]])  # 平移矩阵
A = np.dot(t, s)  # 矩阵相乘
print("A:", A)

# A: [[  0.5   0.  100. ]
#  [  0.    0.5 200. ]
#  [  0.    0.    1. ]]



# 注：等比例缩放的仿射变换,cv2函数
# cv2.getRotationMatrix2D(center, angle, scale)  # center:变换中心点的坐标, angle: 逆时针旋转的角度(单位:角度), scale: 等比例缩放的系数

# 例：计算以坐标点(40， 50)为中心逆时针旋转30°的仿射变换矩阵
A = cv2.getRotationMatrix2D(center=(40, 50), angle=30, scale=0.5)
print("A:", A)
print("A_dtype:", A.dtype)

# A: [[ 0.4330127   0.25       10.17949192]
#  [-0.25        0.4330127  38.34936491]]
# A_dtype: float64