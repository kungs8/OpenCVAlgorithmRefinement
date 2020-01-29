# -*- coding: utf-8 -*- 
# @Time : 2020/1/29 21:25 
# @Author : Yanpeng Gong 
# @Site :  
# @File : 3.1.4计算仿射矩阵(1)_方程法.py 
# @Software: PyCharm


# 例：对空间坐标先等比例缩放2倍，然后在水平方向上平移100， 在垂直方向上平移200，计算仿射变换矩阵代码如下
import numpy as np

s = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])  # 缩放矩阵
t = np.array([[1, 0, 100], [0, 1, 200], [0, 0, 1]])  # 平移矩阵
A = np.dot(t, s)  # 矩阵相乘
print("A:", A)

# A: [[0.5 0.  0. ]
#  [0.  0.5 0. ]]