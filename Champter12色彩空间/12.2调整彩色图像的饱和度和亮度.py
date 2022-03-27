# -*- encoding: utf-8 -*-
'''
@File    :   12.2调整彩色图像的饱和度和亮度.py
@Time    :   2022/03/27 17:53:53
@Author  :   yanpenggong
@Version :   1.0
@Email   :   yanpenggong@163.com
@Copyright : 侵权必究
'''

# here put the import lib

# ----------------------------------------------------------------------------------------------------------------------
# 因为在HLS或HSV色彩空间中都将饱和度和亮度单独分离出来，所以首先将RGB图像转换为HLS或HSV图像，然后调整饱和度和亮度分量，最后将调整后的HLS或者HSV图像转换为RGB图像。

# ----------------------------------------------------------------------------------------------------------------------
# 首先将RGB图像值归一化道[0, 1]，然后使用函数 cvtColor 进行色彩空间的转换，
# 接下来可以根据处理灰度图像对比度增强的伽马变换或者线性变换调整饱和度和亮度分量，最后再转换到RGB色彩空间。
import sys
import cv2
import numpy as np


def main():
    # 判断运行是否正确
    if len(sys.argv) > 1:
        image = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    else:
        print("Usage: python ***.py imageFile")
    
    # 显示原图
    cv2.imshow("image", image)

    # 图像归一化，且转换为浮点型
    fImg = image.astype(np.float32)
    fImg = fImg / 255.0

    # 色彩空间转换
    hlsImg = cv2.cvtColor(fImg, cv2.COLOR_BGR2HLS)
    l, s = 0, 0
    MAX_VALUE = 100
    cv2.namedWindow(winname="l and s",flags=cv2.WINDOW_AUTOSIZE)
    def nothing(*arg):
        pass
    # 绑定滑动条和窗口，定义滚动条的数值
    cv2.createTrackbar("l", "l and s", l, MAX_VALUE, nothing)
    cv2.createTrackbar("s", "l and s", l, MAX_VALUE, nothing)
    # 调整饱和度和亮度后的效果
    lsImg = np.zeros(shape=image.shape, dtype=np.float32)
    # 调整饱和度和亮度
    while True:
        # 复制
        hlsCopy = np.copy(hlsImg)
        # 得到 l 和 s 的值
        l = cv2.getTrackbarPos(trackbarname="l", winname="l and s")
        s = cv2.getTrackbarPos(trackbarname="s", winname="l and s")
        # 调整博啊合度和亮度(线性变换)
        hlsCopy[:, :, 1] = (1.0 + l/float(MAX_VALUE)) * hlsCopy[:, :, 1]
        hlsCopy[:, :, 1][hlsCopy[:, :, 1] > 1] = 1
        hlsCopy[:, :, 2] = (1.0 + s/float(MAX_VALUE)) * hlsCopy[:, :, 2]
        hlsCopy[:, :, 2][hlsCopy[:, :, 2] > 1] = 1
        # HLS2BGR
        lsImg = cv2.cvtColor(src=hlsCopy, code=cv2.COLOR_HLS2BGR)
        # 显示调整后的效果
        cv2.imshow(winname="l and s", mat=lsImg)
        ch = cv2.waitKey(5)
        if ch == 27:
            break
    cv2.destroyAllWindows()



if __name__ == '__main__':
    #运行： python ./Champter12色彩空间/12.2调整彩色图像的饱和度和亮度.py test_images/第12章/img1.jpg
    main()
    
# 结果显示了通过现象变换 y = ax + b 的方式，调整饱和度和亮度后的效果。

# ----------------------------------------------------------------------------------------------------------------------
# 以上实现给出了处理彩色图像的大致步骤。
# 前面Champter 中 都是以处理灰度图像为例的，
# 处理彩色图像的其他方式，如 图像平滑、频率域滤波、形态学处理等，与上述调整饱和度和亮度的步骤类似，
# 往往会根据不同的问题，先将RGB图像转换到其他彩色空间，并分离出每一个通道，然后对每一个通道进行处理，
# 接下来合并，最后再转换为RGB图像，从而实现RGB图像的数字化处理。