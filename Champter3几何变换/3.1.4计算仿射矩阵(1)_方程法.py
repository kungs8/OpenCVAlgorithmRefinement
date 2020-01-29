# -*- coding: utf-8 -*- 
# @Time : 2020/1/29 21:25 
# @Author : Yanpeng Gong 
# @Site :  
# @File : 3.1.4计算仿射矩阵(1)_方程法.py 
# @Software: PyCharm

# 例：
# 如果(0, 0)、(200, 0)、(0, 200)这三个坐标通过某仿射变换矩阵A分别转换为(0, 0)、(100, 0)、(0, 100),则可利用这三组对应坐标构造出6个方程，求解出A

import cv2
import numpy as np

src = np.array([[0, 0], [200, 0], [0, 200]], np.float32)
dst = np.array([[0, 0], [100, 0], [0, 100]], np.float32)
A = cv2.getAffineTransform(src, dst)  # src: 源坐标点矩阵  dst: 结果坐标点矩阵
print("A:", A)

# A: [[0.5 0.  0. ]
#  [0.  0.5 0. ]]