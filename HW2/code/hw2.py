import os
import cv2
import numpy as np
from PIL import Image
import math

def readImage(filename):
    # 读入图像,得到原始图像矩阵
    img = cv2.imread(filename)

    # 转为灰度图像
    gray = np.zeros([512, 512])
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = gray.astype('float64')

    # 图像大小
    h = img.shape[0]
    w = img.shape[1]
    return img, gray, h, w


#计算梯度幅值和方向角
def gradient(gray,h,w,name):
    Gx = np.zeros([h, w], dtype='float64')
    Gy = np.zeros([h, w], dtype='float64')
    M = np.zeros([h, w], dtype='float64')
    theta = np.zeros([h, w], dtype='float64')

    # 运用sobel算子计算梯度幅值和方向角
    Gx = gray.copy()
    Gy = gray.copy()
    kernal_x = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]), dtype='float64')
    kernal_y = np.array(([1, 2, 1], [0, 0, 0], [-1, -2, -1]), dtype='float64')
    for i in range(1,h-1):
        for j in range(1,w-1):
            Gx[i,j] = np.sum(kernal_x * gray[i - 1:i + 2, j - 1: j + 2])
            Gy[i,j] = np.sum(kernal_y * gray[i - 1:i + 2, j - 1:j + 2])

    # 梯度幅值
    M = np.sqrt(Gx ** 2 + Gy ** 2)
    # 方向角
    theta = np.arctan2(Gy, Gx)
    return M, theta


#Non-Max Supression
def nms(M,theta,h,w,name):
    N = M.copy()
    angle = np.zeros([h, w], dtype='float64')

    for i in range(1, M.shape[0] - 1):
        for j in range(1, M.shape[1] - 1):
            angle[i, j] = round(math.degrees(theta[i, j]), 1)
            if ((angle[i, j] >= -22.5 and angle[i, j] <= 22.5) or angle[i, j] >= 157.5 or angle[i, j] <= -157.5):  # 0
                pixel1 = M[i, j + 1]
                pixel2 = M[i, j - 1]
            elif ((angle[i, j] > 22.5 and angle[i, j] <= 67.5) or (
                    angle[i, j] > -157.5 and angle[i, j] <= -112.5)):  # 1
                pixel1 = M[i + 1, j - 1]
                pixel2 = M[i - 1, j + 1]
            elif ((angle[i, j] > 67.5 and angle[i, j] <= 112.5) or (
                    angle[i, j] > -112.5 and angle[i, j] <= -67.5)):  # 2
                pixel1 = M[i + 1, j]
                pixel2 = M[i - 1, j]
            elif ((angle[i, j] > 112.5 and angle[i, j] < 157.5) or (angle[i, j] > -67.5 and angle[i, j] < -22.5)):  # 3
                pixel1 = M[i + 1, j + 1]
                pixel2 = M[i - 1, j - 1]
            if (M[i, j] != max(M[i, j], pixel1, pixel2)):
                N[i, j] = 0

    return N

#双阈值化并边缘链接
def doubleThreshold(N,h,w,high,low,name):
    high_t = np.zeros([h, w], dtype='uint8')
    low_t = np.zeros([h, w], dtype='uint8')

    for i in range(h):
        for j in range(w):
            if N[i, j] >= high:
                high_t[i, j] = 255
            else:
                high_t[i, j] = 0

    for i in range(h):
        for j in range(w):
            if N[i, j] >= low:
                low_t[i, j] = 255
            else:
                low_t[i, j] = 0

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if high_t[i, j] == 255:
                for x in range(i - 1, i + 2):
                    for y in range(j - 1, j + 2):
                        if low_t[x, y] == 255:
                            high_t[x, y] = 255

    return high_t
# mask
def mask(img, mask):
    final = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)
    return final

def Canny(filename,name):
    #读入图像
    img, gray, h, w = readImage(filename)

    # 高斯模糊
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    #用一阶偏导有限差分计算梯度幅值和方向角
    M, theta = gradient(gray,h,w,name)
    M.astype('uint8')
    cv2.imwrite('1-' + name + '-gradient.bmp', M)
    M.astype('float64')

    # Non-Max Supression
    N = nms(M,theta,h,w,name)
    N.astype('uint8')

    cv2.imwrite('2-' + name + '-non-max supression.bmp', N)

    # 双阈值化并边缘连接100-50
    high_t = doubleThreshold(N,h,w,100,50,name)
    cv2.imwrite('3-' + name +'100-50-' + 'double threshold.bmp', high_t)
    # 最终边缘结果图覆盖在原彩色图像上
    final = mask(img, high_t)
    cv2.imwrite('4-' + name + '100-50-' + 'final.bmp', final)

    # 双阈值化并边缘连接200-50
    high_t = doubleThreshold(N, h, w, 200, 50, name)
    cv2.imwrite('3-' + name + '200-50-' + 'double threshold.bmp', high_t)
    # 最终边缘结果图覆盖在原彩色图像上
    final = mask(img, high_t)
    cv2.imwrite('4-' + name + '200-50-' + 'final.bmp', final)

    # 双阈值化并边缘连接100-20
    high_t = doubleThreshold(N, h, w, 100, 20, name)
    cv2.imwrite('3-' + name + '100-20-' + 'double threshold.bmp', high_t)
    #最终边缘结果图覆盖在原彩色图像上
    final = mask(img,high_t)
    cv2.imwrite('4-' + name + '100-20-'+ 'final.bmp',final)


def main():
    filename = 'img/lena.bmp'
    name = 'lena'
    Canny(filename,name)

if __name__=="__main__":
    main()
