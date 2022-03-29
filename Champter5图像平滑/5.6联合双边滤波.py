# -*- encoding: utf-8 -*-
'''
@File    :   5.6联合双边滤波.py
@Time    :   2022/03/29 21:30:50
@Author  :   yanpenggong
@Version :   1.0
@Email   :   yanpenggong@163.com
@Copyright : 侵权必究
'''

# here put the import lib
# ----------------------------------------------------------------------------------------------------------------------
# 5.6.1 原理详解
# 联合双边滤波(Join bilaterral Filter 或称 Cross Bilater Filter) 与双边滤波类似，具体过程如下：
#     Step 1. 对每个位置的邻域构建空间距离权重模板。与双边滤波构建空间距离权重模板一样。
#     Step 2. 构建相似性权重模板。这是与双边滤波唯一的不同之处，双边滤波是根据原图，对于每一个位置，通过该位置和其邻域的灰度值的差的指数来估计相似性；
#             而联合双边滤波是首先对原图进行高斯平滑，根据平滑的结果，用当前位置及其邻域的值的差来估计相似性权重模板。
#     Step 3. 空间距离权重模板和相似性权重模板点乘，然后归一化，作为最后的权重模板。
#     Step 4. 最后将权重模板与原图(注意不是高斯平滑的结果)在该位置的邻域对应位置积的和作为输出值。
#     整个过程只在第二步计算相似性权重模板时和双边滤波不同，但是对图像平滑的效果，特别是对纹理图像来说，却有很大的不同。

# ----------------------------------------------------------------------------------------------------------------------
# 5.6.2 python 实现
# 通过定义函数 joinBLF 实现联合双边滤波，其中构建空间距离权重模板的函数 getClosenessWeight 和双边滤波是一样的，
# 参数 I 代表输入矩阵，注意这里不需要像双边滤波那样进行灰度值归一化；
#     - H、W 分别代表权重模板的高和宽，两者均为奇数；
#     - sigma_g 和 sigma_d 分别代表空间距离权重模板和相似性权重模板的标准差，
# 这四个参数和双边滤波的定义是一样的。
# 在双边滤波的实现代码中，并没有像卷积平滑那样对边界进行扩充，需要在代码中判断边界，为了省去判断边界的问题，在联合双边滤波的实现中对矩阵进行边界扩充操作，
# 即参数 borderType 对含义，对于扩充边界的处理，这一点就类似于 OpenCV实现的双边滤波。代码如下：
import sys
import numpy as np
import math
import cv2

def getClosenessWeight(sigma_g, H, W):
    """
    定义函数getClosenessWeight 构建 H * W 的空间距离权重和构建高斯卷积核类似
    :param sigma_g: 空间距离权重模板的标准差
    :param H: 权重模板的高(数值为奇数)
    :param W: 权重模板的宽(数值为奇数)
    :return:
    """
    r, c = np.mgrid[0:H:1, 0:W:1]
    r -= int((H-1) / 2)
    c -= int((W-1) / 2)
    closeWeight = np.exp(-0.5 * (np.power(r, 2) + np.power(c, 2)) / math.pow(sigma_g, 2))
    return closeWeight

def joinBLF(I, H, W, sigma_g, sigma_d, borderType=cv2.BORDER_DEFAULT):
    # 构建空间距离权重模板
    closenessWeight = getClosenessWeight(sigma_g, H, W)
    # 对 I 进行高斯平滑
    Ig = cv2.GaussianBlur(src=I, ksize=(W, H), sigmaX=sigma_g)
    # 模板的中心点位置
    cH = int((H - 1) / 2)
    cW = int((W - 1) / 2)
    # 对原图和高斯平滑的结果扩充边界
    Ip = cv2.copyMakeBorder(src=I, top=cH, bottom=cH, left=cW, right=cW, borderType=borderType)
    Igp = cv2.copyMakeBorder(src=Ig, top=cH, bottom=cH, left=cW, right=cW, borderType=borderType)
    # 图像矩阵的行数和列数
    rows, cols = I.shape
    i, j = 0, 0
    # 联合双边滤波的结果
    jblf = np.zeros(I.shape, np.float64)
    for r in np.arange(cH, cH+rows, 1):
        for c in np.arange(cW, cW+cols, 1):
            # 当前位置的值
            pixel = Igp[r][c]
            # 当前位置的邻域
            rTop, rBottom = r - cH, r + cH
            cLeft, cRight = c - cW, c + cW
            # 从 Igp 中截取该邻域，用于构建相似性权重模板
            region = Igp[rTop:rBottom+1, cLeft:cRight+1]
            # 通过上述邻域，构建该位置的相似性权重模板
            silimarityWeight = np.exp(-0.5*np.power(region - pixel, 2.0) / math.pow(sigma_d, 2.0))
            # 相似性权重模板和空间距离全职模板相乘
            weight = closenessWeight * silimarityWeight
            # 将权重模板归一化
            weight = weight / np.sum(weight)
            # 权重模板和邻域对应位置相乘并求和
            jblf[i][j] = np.sum(Ip[rTop:rBottom+1, cLeft:cRight+1]*weight)
            j += 1
        j = 0
        i += 1
    return jblf


if __name__ == '__main__':
    if len(sys.argv) > 1:
        I = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    else:
        print("Usage: python **.py imageFile")
    # 将 8 位图转换为浮点型
    fI = I.astype(np.float64)
    # 联合双边滤波，返回值的数据类型为浮点型
    jblf = joinBLF(I=fI, H=33, W=33, sigma_g=7, sigma_d=2)
    # 转换为 8 位图
    jblf = np.round(jblf)
    jblf = jblf.astype(np.uint8)
    cv2.imshow("jblf", jblf)
    # 保存结果
    # cv2.imwrite("jblf1.png", jblf)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 运行
# python 5.6联合双边滤波.py ../OpenCV_ImgData/Champter5/img3.jpg

# ----------------------------------------------------------------------------------------------------------------------
# 进行联合双边滤波处理后，"方形"纹理几乎完全消失，而且同时对边缘的保留也非常好，并没有感觉出边缘模糊的效果。

# 基于双边滤波和联合双边滤波，后面提出了循环引导滤波(Guided Image Filtering), 
# 双边滤波是根据原图计算相似性权重模板的，联合双边滤波对其进行了改进，是根据图像的高斯平滑结果计算相似性权重模板的，
# 而循环引导滤波，顾名思义，是一种迭代的方法，本质上是一种多次迭代的联合双边滤波，
# 只是每次计算相似性权重模板的依据不一样 —— 利用本次计算的联合双边滤波结果作为下一次联合双边滤波计算相似性权重模板的依据。