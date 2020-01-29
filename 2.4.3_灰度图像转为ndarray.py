# -*- coding: utf-8 -*- 
# @Time : 2020/1/29 14:21 
# @Author : Yanpeng Gong 
# @Site :  
# @File : 2.4.3_灰度图像转为ndarray.py 
# @Software: PyCharm

import sys
import cv2
import numpy as np

if __name__ == '__main__':
    # 输入图像矩阵，转换为array
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.CV_IMAGE_GRAYSCALE)
    else:
        print("Usge: python imgToArray.py imageFile")
    # 显示图像
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()