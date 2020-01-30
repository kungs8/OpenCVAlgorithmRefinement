# -*- coding: utf-8 -*- 
# @Time : 2020/1/30 16:09 
# @Author : Yanpeng Gong 
# @Site :  
# @File : 3.1.8旋转函数rotate.py 
# @Software: PyCharm
import sys

import cv2

# cv2.rotate(src, rotateCode, dst)
# src: 输入矩阵(单、多通道矩阵都可以)
# rotateCode: ROTATE_90_CLOCKWISE(顺时针旋转90°); ROTATE_180(顺时针旋转180°); ROTATE_90_COUNTERCLOCKWISE(顺时针旋转270°)
# dst: 输出矩阵
# 注:这里的图像矩阵旋转，不需要利用仿射变换来完成这类旋转，只是行列的互换，类似于矩阵的转置操作

if __name__ == '__main__':
    if len(sys.argv) > 1:
        image = cv2.imread(sys.argv[1], cv2.IMREAD_ANYCOLOR)
    else:
        print("Usage: python rotate.py image")

    # 显示原图
    cv2.imshow("image", image)

    # 图像旋转
    rImg90 = cv2.rotate(src=image, rotateCode=cv2.ROTATE_90_CLOCKWISE)
    rImg180 = cv2.rotate(src=image, rotateCode=cv2.ROTATE_180)
    rImg270 = cv2.rotate(src=image, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
    # 显示旋转的结果
    cv2.imshow("rImg90", rImg90)
    cv2.imshow("rImg180", rImg180)
    cv2.imshow("rImg270", rImg270)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 运行
# ipython 3.1.8旋转函数rotate.py ../OpenCV_ImgData/Champter3/img.jpg