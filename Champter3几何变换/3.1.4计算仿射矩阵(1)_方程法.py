# -*- coding: utf-8 -*- 
# @Time : 2020/1/29 21:25 
# @Author : Yanpeng Gong 
# @Site :  
# @File : 3.1.4计算仿射矩阵(1)_方程法.py 
# @Software: PyCharm

import cv2
import numpy as np

src = np.array([[0, 0], [200, 0], [0, 200]], np.float32)
dst = np.array([[0, 0], [100, 0], [0, 100]], np.float32)
A = cv2.getAffineTransform(src, dst)
print("A:", A)

# A: [[0.5 0.  0. ]
#  [0.  0.5 0. ]]