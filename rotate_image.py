# Copyright 2020 chenli Authors. All Rights Reserved.
# This file in order to  rotate rectangle object image
# 该文件是为了旋转有长方形目标的图像，校正为水平方向，只能水平校正，不能三维校正。
import numpy as np
import cv2
import math
from scipy import ndimage
is_print = False


def rotate_image(img):
    """ 获得出现最多次的线作为图像的方向
    image(Mat): image
    :return
        rhoLevelAll[index]: rho,
        slopeLevelAll[index]: theta
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 霍夫变换
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 130, 40, 0)
    # print(lines)
    # print(lines[0])
    if len(lines) == 0:
        return image

    rho, theta = get_max_line(lines)
    # print(rho, theta)
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    if x2 == x1:
        return img
    t = float(y2 - y1) / (x2 - x1)
    rotate_angle = math.degrees(math.atan(t))
    if rotate_angle > 45:
        rotate_angle = -90 + rotate_angle
    elif rotate_angle < -45:
        rotate_angle = 90 + rotate_angle
    rotate_img = ndimage.rotate(img, rotate_angle, cval=255.0)

    return rotate_img


def get_max_line(lines):
    """ 获得出现最多次的线作为图像的方向
    image(Mat): image
    line(list): list(rho, theta)
    :return
        rhoLevelAll[index]: rho,
        slopeLevelAll[index]: theta
    """
    slopeLevelAll, rhoLevelAll, LevelAll_num = getAllSlopePoint(lines, Threshold=0.1)
    if is_print:
        print("slopeLevel All", slopeLevelAll)
        print("LevelAll num", LevelAll_num)
    maxs = -1
    index = 0
    for i, number in enumerate(LevelAll_num):
        if abs(slopeLevelAll[i] - 1.5707964) < 0.02:
            return rhoLevelAll[i], slopeLevelAll[i]
        if number > maxs:
            maxs = number
            index = i
    return rhoLevelAll[index], slopeLevelAll[index]


def getAllSlopePoint(lines, Threshold=0.1):
    """ 获得斜率频域数组
    line(list): list(rho, theta)
    Threshold: if slope difference min than Threshold, they are similarity
    :return
        rhoLevelAll(list): rho
        slopeLevelAll(list): theta
        LevelAll_num(list): some slope number
    """
    # 将检测的线画出来,极坐标
    slopeLevelAll = []
    rhoLevelAll = []
    LevelAll_num = []

    for line in lines:
        rho, slope = line[0]

        # 判断是否为斜线
        if slope < np.pi / 3.0 or slope > np.pi - np.pi / 3.0:
            continue

        # 判断是否为斜率相近的斜线，相近的在其数量上加一
        flags = 0
        for i in range(len(rhoLevelAll)):
            if (abs(slope - slopeLevelAll[i]) < Threshold or abs(
                    slope - slopeLevelAll[i]) > np.pi - Threshold):
                flags = 1
                LevelAll_num[i] = LevelAll_num[i] + 1
                break

        if flags == 1:
            continue

        slopeLevelAll.append(slope)
        rhoLevelAll.append(rho)
        LevelAll_num.append(1)

    return slopeLevelAll, rhoLevelAll, LevelAll_num

if __name__ == '__main__':

    filepath = 'image/rotate_image2.jpg'

    import time

    image = cv2.imread(filepath)
    start = time.time()
    imgs = rotate_image(image)
    rotate_time = time.time()
    print("rotate_time:", rotate_time - start)
    cv2.imshow("image", imgs)
    cv2.waitKey()
    cv2.destroyAllWindows()
