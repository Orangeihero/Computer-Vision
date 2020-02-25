import os
import cv2
import numpy as np


# 得到图片的特征值点
def getFeatures(image):
    sift = cv2.xfeatures2d.SIFT_create()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)  # 转为灰度图像
    kp, des = sift.detectAndCompute(gray, None)  # 检测关键点，计算描述符，kp是关键点列表，des是形状为Number_of_Keypoints×128的numpy数组
    return kp, des

# 得到中间图像（两张图象对应的特征点连线图）
def getMatchImage(previous_image, current_image, file_name):
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)  # 指定递归遍历的次数checks，值越高越准确，消耗的时间也就越多
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    kp1, des1 = getFeatures(previous_image)
    kp2, des2 = getFeatures(current_image)

    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)
    good = np.expand_dims(good, 1)
    t = int(len(good)/2)

    img = cv2.drawMatchesKnn(previous_image, kp1, current_image, kp2, good, None, flags=2)
    cv2.imwrite(file_name, img)

    previous = previous_image.copy()
    previous = cv2.drawKeypoints(previous,kp1,previous)
    current = current_image.copy()
    current = cv2.drawKeypoints(current,kp2,current)
    img1 = cv2.drawMatchesKnn(previous, kp1, current, kp2, good, None, flags=2)
    if file_name[0] == 'i':
        file_name_line = file_name.replace('image','image_line_point')
    else:
        file_name_line = file_name.replace('stitch','stitch_line_point')
    cv2.imwrite(file_name_line, img1)


# 匹配两张图片，得到H矩阵
def match(previous_image, current_image, file_name):
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)  # 指定递归遍历的次数checks，值越高越准确，消耗的时间也就越多
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    kp1, des1 = getFeatures(previous_image)
    kp2, des2 = getFeatures(current_image)

    #获得两张图片匹配拼接的过程图
    getMatchImage(previous_image.copy(), current_image.copy(), file_name)
    file_name1=file_name.replace('.jpg','_(1).jpg')
    file_name1 = file_name1.replace('stitch', 'stitch_feature')
    previous = previous_image.copy()
    previous = cv2.drawKeypoints(previous,kp1,previous)
    cv2.imwrite(file_name1,previous)

    file_name2 = file_name.replace('.jpg', '_(2).jpg')
    file_name2 = file_name2.replace('stitch', 'stitch_feature')
    current = current_image.copy()
    current = cv2.drawKeypoints(current, kp2,current)
    cv2.imwrite(file_name2, current)

    #进行匹配，得到单应性矩阵
    matches = flann.knnMatch(des2, des1, k=2)
    good = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good.append((m.trainIdx, m.queryIdx))

    if len(good) > 4:
        current_points = np.float32(
            [kp2[i].pt for (__, i) in good]
        )
        previous_points = np.float32(
            [kp1[i].pt for (i, __) in good]
        )
        H, _ = cv2.findHomography(current_points, previous_points, cv2.RANSAC, 4)
        return H
    return None

#得到H矩阵
def getHomography(previous_image, current_image, i):
    file_name = 'stitch' + '_' + str(i) + "_" + str(i + 1) + '.jpg'
    H = match(previous_image, current_image, file_name)
    return H

#处理图像拼接留下的分割线
def blendImage(temp, result, start_x, start_y, w, h, direction):
    start_x = int(start_x)
    start_y = int(start_y)
    h = int(h)
    w = int(w)
    for i in range(start_y, start_y + h):
        for j in range(start_x, start_x + w):
            if (result[i, j, 0] == 0 and result[i, j, 1] == 0 and result[i, j, 2] == 0):
                result[i, j, :] = temp[i, j, :]
            elif (temp[i, j, 0] == 0 and temp[i, j, 1] == 0 and temp[i, j, 2] == 0):
                result[i, j, :] = result[i, j, :]
            else:
                alpha = (w - (j - start_x)) / w
                if (direction == 'left'):
                    result[i, j, 0] = alpha * temp[i, j, 0] + (1 - alpha) * result[i, j, 0]
                    result[i, j, 1] = alpha * temp[i, j, 1] + (1 - alpha) * result[i, j, 1]
                    result[i, j, 2] = alpha * temp[i, j, 2] + (1 - alpha) * result[i, j, 2]
                else:
                    result[i, j, 0] = alpha * result[i, j, 0] + (1 - alpha) * temp[i, j, 0]
                    result[i, j, 1] = alpha * result[i, j, 1] + (1 - alpha) * temp[i, j, 1]
                    result[i, j, 2] = alpha * result[i, j, 2] + (1 - alpha) * temp[i, j, 2]
    return result

#得到最终输出的图像的大小
def getSize(temp, current_image, offset_x, offset_y, ds, dsize):
    up = offset_y
    down = offset_y + current_image.shape[0]
    left = 0

    if np.array_equal(temp[offset_y, offset_x], [0, 0, 0]):
        for i in range(offset_y, int(ds[1])):
            if not np.array_equal(temp[i, offset_x],[0, 0, 0]):
                up = i
                break

    for i in range(dsize[1]):
        if not np.array_equal(temp[up, i], [0, 0, 0]):
            left = i
            break

    if np.array_equal(temp[current_image.shape[0] + offset_y, offset_x], [0, 0, 0]):
        for i in range(int(ds[1]), current_image.shape[0] + offset_y):
            if not np.array_equal(temp[i, offset_x], [0, 0, 0]):
                down = i
                break

    for i in range(dsize[1]):
        if not np.array_equal(temp[down, i], [0, 0, 0]):
            left = max(left, i)
            break

    return up, down, left

# 对于left_part的拼接
def leftStitch(left_part):
    global up, down,left,right
    previous_image = left_part[0]
    for i, current_image in enumerate(left_part[1:]):
        H = getHomography(previous_image.copy(), current_image.copy(), i+1)
        IH = np.linalg.inv(H)  # 矩阵求逆

        offset = np.dot(IH, np.array([0, 0, 1]))  # 矩阵积
        offset = offset / offset[-1]  # 规范化

        # 改变矩阵，使其可以显示在画布上
        IH[0][-1] += abs(offset[0])
        IH[1][-1] += abs(offset[1])

        ds = np.dot(IH, np.array([previous_image.shape[1], previous_image.shape[0], 1]))
        offset_y = abs(int(offset[1]))
        offset_x = abs(int(offset[0]))
        dsize = (offset_x + current_image.shape[1], int(ds[1]) + offset_y)

        temp = cv2.warpPerspective(previous_image, IH, dsize, borderMode=cv2.BORDER_TRANSPARENT)
        # print(temp.shape)
        result = temp.copy()
        result[offset_y: current_image.shape[0] + offset_y, offset_x: current_image.shape[1] + offset_x] = current_image


        result = blendImage(temp, result, offset_x, offset_y, 50, ds[1] - offset_y, 'left')
        previous_image = result.copy()

    up,down,left = getSize(temp,current_image,offset_x,offset_y,ds,dsize)
    return result

#对于right_part中图像两两拼接的处理
def mix_and_match(left_image, warped_image):
    y, x = left_image.shape[:2]

    for i in range(0, x):
        for j in range(0, y):
            try:
                if (np.array_equal(left_image[j, i], np.array([0, 0, 0])) and np.array_equal(warped_image[j, i],
                                                                                            np.array([0, 0, 0]))):
                    warped_image[j, i] = [0, 0, 0]
                else:
                    if (np.array_equal(warped_image[j, i], [0, 0, 0])):
                        warped_image[j, i] = left_image[j, i]
                    else:
                        if not np.array_equal(left_image[j, i], [0, 0, 0]):
                            bl, gl, rl = left_image[j, i]
                            warped_image[j, i] = [bl, gl, rl]
            except:
                pass
    return warped_image

#对于right_part的拼接
def rightStitch(panorama, right_part, num):
    previous_image = panorama

    for i, current_image in enumerate(right_part):
        H = getHomography(previous_image, current_image, num + i)
        ds = np.dot(H, np.array([current_image.shape[1], current_image.shape[0], 1]))
        ds = ds / ds[-1]
        dsize = (int(ds[0]), max(int(ds[1]), previous_image.shape[0]))
        temp = cv2.warpPerspective(current_image, H, dsize)
        result = mix_and_match(previous_image, temp.copy())
        result = blendImage(temp, result, previous_image.shape[1] - 100, 0, 100, previous_image.shape[0], 'right')
        previous_image = result.copy()

    down_new = down
    for i in range(result.shape[0]-1,0,-1):
        if(not np.array_equal(result[i, result.shape[1]-5], np.array([0, 0, 0]))):
            down_new =min(i,down_new)
            break

    right = result.shape[1]
    for i in range(result.shape[1]-1,0,-1):
        if(not np.array_equal(result[0,i], np.array([0, 0, 0]))):
            right =min(i,right)
            break

    return result[up:down_new,left:right]


#将图像列表分为左右两部分
def divide(images):
    left_part = []
    right_part = []

    center = int(len(images) / 2)

    for i in range(len(images)):
        if i <= center:
            left_part.append(images[i])
        else:
            right_part.append(images[i])

    return left_part, right_part


def main():

    # 图像列表
    images = [
        "images/yosemite/yosemite1.jpg",
        "images/yosemite/yosemite2.jpg",
        "images/yosemite/yosemite3.jpg",
        "images/yosemite/yosemite4.jpg"
    ]

    # images = [
    #     "images/tree/tree1.jpg",
    #     "images/tree/tree2.jpg",
    #     "images/tree/tree3.jpg",
    #     "images/tree/tree4.jpg",
    #     "images/tree/tree5.jpg",
    #     "images/tree/tree6.jpg"
    # ]

    # images = [
    #     "images/building/building1.jpg",
    #     "images/building/building2.jpg",
    #     "images/building/building3.jpg",
    #     "images/building/building4.jpg",
    #     "images/building/building5.jpg"
    # ]

    #重新定义图像的大小
    for i in range(len(images)):
        images[i] = cv2.resize(cv2.imread(images[i]), (800, 450))

    #得到原图像的特征点和两张图像之间的对应点连线
    for i in range(len(images)):
        if i < len(images) - 1:
            file_name = 'image' + '_' + str(i+1) + '_' + str(i + 2) + '.jpg'
            getMatchImage(images[i], images[i + 1], file_name)
        kp, des = getFeatures(images[i])
        image_feature = images[i].copy()
        image_feature = cv2.drawKeypoints(image_feature,kp,image_feature)
        cv2.imwrite('image_feature_'+str(i+1)+'.jpg',image_feature)

    #将图像列表分为左右两部分
    left_part, right_part = divide(images)

    # 对左边部分进行拼接
    panorama = leftStitch(left_part)
    #对右边部分进行拼接
    panorama = rightStitch(panorama, right_part, len(left_part))
    cv2.imwrite('final.jpg', panorama)


if __name__ == "__main__":
    main()
