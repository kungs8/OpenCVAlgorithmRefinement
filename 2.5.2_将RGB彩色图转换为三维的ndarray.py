# -*- coding: utf-8 -*- 
# @Time : 2020/1/29 16:15 
# @Author : Yanpeng Gong 
# @Site :  
# @File : 2.5.2_将RGB彩色图转换为三维的ndarray.py 
# @Software: PyCharm

import cv2
import numpy as np
import sys

if __name__ == '__main__':
    if len(sys.argv) >1:
        image = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    else:
        print("Usge: python RGB.py imageFile")
    # 得到三个颜色的通道
    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]

    # 显示三个颜色的通道
    cv2.imshow("b", b)
    cv2.imshow("g", g)
    cv2.imshow("r", r)
    cv2.waitKey(0)  # 这里delaytime=0, 表示永不退出，除非按"Esc"键. delaytime单位：ms
    cv2.destroyAllWindows()