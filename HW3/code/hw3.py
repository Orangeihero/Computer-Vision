import cv2
import numpy as np
import math

def readImage(filename,type):
    # 读入图片，并转为灰度图片
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    if type == 'f':
        gray = gray.astype('float64')

    #得到图片的尺寸
    h = img.shape[0]
    w = img.shape[1]
    return img,gray,h,w

#计算图像X方向与Y方向的一阶高斯偏导数及其相关矩阵
def getIxy(gray,h,w):
    Ix = np.zeros([h, w])
    Iy = np.zeros([h, w])

    Ix = cv2.Sobel(gray, -1, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, -1, 0, 1, ksize=3)

    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    # 对Ix^2，Iy^2,Ix*Iy进行高斯卷积
    kernal = np.array(([1, 2, 1], [2, 4, 2], [1, 2, 1]), dtype='float64')
    kernal *= 1.0 / 16.0
    Sxx = cv2.filter2D(Ix2, -1, kernal)
    Syy = cv2.filter2D(Iy2, -1, kernal)
    Sxy = cv2.filter2D(Ixy, -1, kernal)
    return Sxx,Syy,Sxy

#得到R，最大特征值矩阵和最小特征值矩阵
def getR_Lam(Sxx,Syy,Sxy,h,w):
    # 计算R
    R = np.zeros([h, w], dtype='float64')
    M = np.zeros([2, 2], dtype='float64')
    lam_max = np.zeros([h, w], dtype='float64')
    lam_min = np.zeros([h, w], dtype='float64')
    for i in range(h):
        for j in range(w):
            M[0, 0] = Sxx[i, j]
            M[0, 1] = Sxy[i, j]
            M[1, 0] = Sxy[i, j]
            M[1, 1] = Syy[i, j]

            #计算M的特征值
            e, f = np.linalg.eig(M)
            [lam1, lam2] = e

            #计算R
            detM = lam1 * lam2
            traceM = lam1 + lam2
            R[i, j] = detM - 0.05 * (traceM ** 2)

            #得到最大特征向量和最小特征向量
            if(lam1 >= lam2):
                lam_max[i,j] = lam1
                lam_min[i,j] = lam2
            else:
                lam_max[i, j] = lam2
                lam_min[i, j] = lam1

    return R,lam_max,lam_min

#设置阈值得到图像的边和角点
def getCornerEdge(R,h,w):
    corner = np.zeros([h,w],dtype='uint8')
    edge = np.zeros([h,w],dtype='uint8')

    #得到矩阵的最大值和最小值
    max = np.max(R)
    min = np.min(R)

    # 设置阈值
    #得到图像的角点
    for i in range(h):
        for j in range(w):
            if R[i,j] > 0:
                if R[i,j] > max * 0.001:
                    corner[i,j] = 255

    #得到图像的边
    for i in range(h):
        for j in range(w):
            if R[i,j] < 0:
                if abs(R[i,j]) > abs(min*0.001):
                    edge[i,j] = 255

    cv2.imwrite('./img/corner.jpg',corner)
    cv2.imwrite('./img/edge.jpg',edge)

#得到最大特征图和最小特征图
def getMaxMin(lam_max,lam_min,h,w):
    max1 = np.zeros([h, w], dtype='float64')
    min1 = np.zeros([h, w], dtype='float64')

    #归一化
    cv2.normalize(lam_max, max1, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(lam_min, min1, 0, 255, cv2.NORM_MINMAX)

    min1 = min1.astype(np.uint8)
    max1 = max1.astype(np.uint8)
    cv2.imwrite('./img/lam_max.jpg', max1)
    cv2.imwrite('./img/lam_min.jpg', min1)


#得到热力学图
def getRimg(R,h,w):
    n_R = np.zeros([h,w])
    cv2.normalize(R,n_R,0,255,cv2.NORM_MINMAX)
    n_R = n_R.astype(np.uint8)
    heat_img = cv2.applyColorMap(n_R, cv2.COLORMAP_JET)  # 此处的三通道热力图是cv2专有的BGR排列
    heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)  # 将BGR图像转为RGB图像
    cv2.imwrite('./img/heat_img.jpg',heat_img)

#原图上叠加显示结果
def nms(img,R,h,w,offset):
    max = np.max(R)
    R[R < max*0.001] = 0

    #筛选检测结果并在原图上绘制
    for i in range(offset,h-offset):
        for j in range(offset,w-offset):
            temp = R[i-offset:i+offset+1,j-offset:j+offset+1]
            if R[i,j] == np.max(temp) and R[i,j] != 0:
                cv2.circle(img, (j,i),2,(0,0,255),-1)

    cv2.imwrite('./img/final.jpg',img)

#Horris Corner Detection检测算法
def harrisConerDetection(filename):
    img,gray,h,w = readImage(filename,'f')
    Sxx,Syy,Sxy = getIxy(gray,h,w)
    R,lam_max, lam_min = getR_Lam(Sxx,Syy,Sxy,h,w)
    getMaxMin(lam_max, lam_min, h, w)
    getCornerEdge(R, h, w)
    nms(img,R,h,w,7)

#得到R图
def Rimg(filename):
    img, gray, h, w = readImage(filename,'u')
    Sxx, Syy, Sxy = getIxy(gray, h, w)
    R,lam_max, lam_min = getR_Lam(Sxx, Syy, Sxy, h, w)
    getRimg(R, h, w)

def main():
    filename = './img/test/original/horse1.jpg'
    harrisConerDetection(filename)
    Rimg(filename)


if __name__=="__main__":
    main()
