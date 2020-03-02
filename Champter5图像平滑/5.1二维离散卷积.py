# -*- coding: utf-8 -*- 
# @Time : 2020/2/26 18:30 
# @Author : Yanpeng Gong 
# @Site :  
# @File : 5.1二维离散卷积.py 
# @Software: PyCharm

# ----------------------------------------------------------------------------------------------------------------------
# 不同的边界扩充方式得到的same卷积只是在距离矩阵上、下、左右四个边界小于卷积核半径的区域内值会不同，所以只要在用卷积运算进行图像处理时，图像的重要信息不要落在距离边界小于卷积核半径的区域内就行。
# 以下是边界扩充方式及函数
# 函数cv2.copyMakeBorder(src, top, bottom, left, right, borderType, dst, value)对矩阵的扩充
    # src：输入矩阵
    # top：上侧扩充的行数
    # bottom：下侧扩充的行数
    # left：左侧扩充的列数
    # right：右侧扩充的列数
    # borderType：
    #     BORDER_REPLICATE(边界复制)
    #     BORDER_CONSTANT(常数扩充)
    #     BORDER_REFLECT(反射扩充)
    #     BORDER_REFLECT_101(以边界为中心反射扩充)
    #     BORDER_WRAP(平铺扩充)
    # dst：输出矩阵
    # value：borderType=BORDER_CONSTANT时的常数
import cv2
import numpy as np

src = np.array([[5, 1, 7], [1, 5, 9], [2, 6, 2]])
dst = cv2.copyMakeBorder(src=src, top=2, bottom=2, left=2, right=2, borderType=cv2.BORDER_REFLECT101)
print("dst:\n", dst)
# dst:
#  [[2 6 2 6 2 6 2]
#  [9 5 1 5 9 5 1]
#  [7 1 5 1 7 1 5]
#  [9 5 1 5 9 5 1]
#  [2 6 2 6 2 6 2]
#  [9 5 1 5 9 5 1]
#  [7 1 5 1 7 1 5]]

# ----------------------------------------------------------------------------------------------------------------------
# 5.1.1 二维离散卷积
# signal.convolve2d(in1, in2, mode='full', boundary='fill', fillvalue=0)
# in1: 输入的二维数组
# in2: 输入的二位数组，代表卷积核
# mode: 卷积类型('full', 'valid', 'same')
# boundary: 边界填充方式('fill', 'wrap', 'symm')
# fillvalue: 当boundary='fill'时, 设置边界填充的值，默认值为0

import numpy as np
from scipy import signal

if __name__ == '__main__':
    # 输入矩阵
    I = np.array([[1, 2], [3, 4]], np.float32)
    # I的高核宽
    H1, W1 = I.shape[:2]
    # 卷积核
    K = np.array([[-1, -2], [2, 1]], np.float32)
    # K的高和宽
    H2, W2 = K.shape[:2]
    # 计算full卷积
    c_full = signal.convolve2d(I, K, mode='full')
    # 指定锚点的位置
    kr, kc = 0, 0
    # 根据锚点的位置，从full卷积中截取得到same卷积
    c_same = c_full[H2 - kr - 1: H1 + H2 - kr - 1, W2 - kc - 1: W1 + W2 - kc - 1]
    print("c_same:\n", c_same)
    # c_same:
    # [[-5. - 6.]
    #  [11.  4.]]

# ----------------------------------------------------------------------------------------------------------------------
# 5.1.2 可分离卷积核
import numpy as np
from scipy import signal

# 主函数
if __name__ == '__main__':
    kernel1 = np.array([[1, 2, 3]], np.float32)
    kernel2 = np.array([[4], [5], [6]], np.float32)
    # 计算两个核的全卷积
    kernel = signal.convolve2d(kernel1, kernel2, mode='full')
    print("kernel:\n", kernel)
    # kernel:
    # [[4.  8. 12.]
    #  [5. 10. 15.]
    # [6. 12. 18.]]

# ----------------------------------------------------------------------------------------------------------------------
# 5.1.3 离散卷积的性质(full卷积)
import numpy as np
from scipy import signal

I = np.array([[1, 2, 3, 10, 12],
              [32, 43, 12, 4, 190],
              [12, 234, 78, 0, 12],
              [43, 90, 32, 8, 90],
              [71, 12, 4, 98, 123]])
kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])
I_Kernel = signal.convolve2d(I, kernel, mode='full', boundary='fill', fillvalue=0)
print("I_Kernel:\n", I_Kernel)
# I_Kernel:
#  [[   1    2    2    8    9  -10  -12]
#  [  33   45  -18  -31  187  -14 -202]
#  [  45  279   48 -265  121  -14 -214]
#  [  87  367   35 -355  170  -12 -292]
#  [ 126  336  -12 -230  111 -106 -225]
#  [ 114  102  -78    4  177 -106 -213]
#  [  71   12  -67   86  119  -98 -123]]

# ----------------------------------------------------------------------------------------------------------------------
# 5.1.3 离散卷积的性质(same卷积)
c_same = signal.convolve2d(I, kernel, mode='same', boundary='fill', fillvalue=0)
print("c_same:\n", c_same)
# c_same:
#  [[  45  -18  -31  187  -14]
#  [ 279   48 -265  121  -14]
#  [ 367   35 -355  170  -12]
#  [ 336  -12 -230  111 -106]
#  [ 102  -78    4  177 -106]]
# ----------------------------------------------------------------------------------------------------------------------
