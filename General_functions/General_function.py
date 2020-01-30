# -*- coding: utf-8 -*- 
# @Time : 2020/1/29 23:13 
# @Author : Yanpeng Gong 
# @Site :  
# @File : General_function.py 
# @Software: PyCharm

import cv2


def CVToLocal(ImgName, img):
    """
    保存到本地
    :param ImgName:  保存到本地的图像名
    :param img:  图像数据
    :return:
    """
    cv2.imwrite(filename="../RstData/{}.jpg".format(ImgName), img=img)
    print("{} save over!".format(ImgName))