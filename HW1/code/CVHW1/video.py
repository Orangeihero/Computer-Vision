import cv2
import os
import numpy as np

#绘制直线
def drawLine(out,img,startPoint,endPoint,color,thick,time,isVertical):
    process = img.copy()
    draw = cv2.line(img.copy(), startPoint, endPoint, color, thick)
    #垂直线
    if(isVertical):
        length = endPoint[1] - startPoint[1]
        if (startPoint[0] < endPoint[0]):
            s_x = startPoint[0]
            e_x = endPoint[0]
        else:
            s_x = endPoint[0]
            e_x = startPoint[0]
        jump = int(length / time)
        for i in range(time):
            temp = draw[(startPoint[1]+jump * i):(startPoint[1]+jump * (i + 1)), (s_x-5):(e_x+5), 0:3]
            process[(startPoint[1]+jump * i):(startPoint[1]+jump * (i + 1)), (s_x-5):(e_x+5), 0:3] = temp
            out.write(process)
    #水平线
    else:
        length = endPoint[0] - startPoint[0]
        if (startPoint[1] < endPoint[1]):
            s_y = startPoint[1]
            e_y = endPoint[1]
        else:
            s_y = endPoint[1]
            e_y = startPoint[1]
        jump = int(length / time)
        for i in range(time):
            temp = draw[(s_y-5):(e_y+5),(startPoint[0] + jump * i):(startPoint[0] + jump * (i + 1)), 0:3]
            process[(s_y-5):(e_y+5),(startPoint[0] + jump * i):(startPoint[0] + jump * (i + 1)), 0:3] = temp
            out.write(process)
    return process

#绘制开头
def drawBegin(out,img):
    image = cv2.imread('photo.jpg')
    image = cv2.resize(image, (400, 400))
    process = img.copy()
    process[100:500, 200:600, 0:3] = image
    info = '317010xxxx, Haha Hahah'

    # 文本的起始坐标
    text_x = 180
    for i in range(66):  # 控制视频的速度
        if (i % 3 == 1):
            j = int(i / 3)
            cv2.putText(process, info[j], (text_x, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
            text_x += 20
        out.write(process)  # 保存一张张图片流，最终形成视频

    return process

#绘制第一幅简笔画
def drawDayNight(out, img):
    process = drawLine(out, img, (400, 0), (400, 600), (0, 0, 0), 1, 20, True)

    # 太阳天空蓝
    for i in range(20):
        process[(30 * i):(30 * i + 30), 0:400, 0] = 231
        process[(30 * i):(30 * i + 30), 0:400, 1] = 192
        process[(30 * i):(30 * i + 30), 0:400, 2] = 93
        out.write(process)

    # 月亮天空黑
    for i in range(20):
        process[(30 * i):(30 * i + 30), 401:800, 0] = 92
        process[(30 * i):(30 * i + 30), 401:800, 1] = 67
        process[(30 * i):(30 * i + 30), 401:800, 2] = 4
        out.write(process)

    # 画太阳
    process = cv2.circle(process, (200,300), 100, (22, 149, 242))
    out.write(process)
    
    draw2 = cv2.ellipse(process.copy(), (200, 300), (100, 100), 0, 0, 360, (22, 149, 242), -1)
    for i in range(20):
        temp3 = np.zeros([200, 10, 3])
        temp3 = draw2[(200 + 10 * i):(210 + 10 * i), 100:300, :]
        process[(200 + 10 * i):(210 + 10 * i), 100:300, :] = temp3
        out.write(process)

    process = drawLine(out, process, (200, 130), (200, 180), (22, 149, 242), 5, 10, True)
    process = drawLine(out, process, (335, 165), (300, 200), (22, 149, 242), 5, 10, True)
    process = drawLine(out, process, (320, 300), (370, 300), (22, 149, 242), 5, 10, False)
    process = drawLine(out, process, (300, 400), (335, 435), (22, 149, 242), 5, 10, True)
    process = drawLine(out, process, (200, 420), (200, 470), (22, 149, 242), 5, 10, True)
    process = drawLine(out, process, (100, 400), (65, 435), (22, 149, 242), 5, 10, True)
    process = drawLine(out, process, (30, 300), (80, 300), (22, 149, 242), 5, 10, False)
    process = drawLine(out, process, (65, 165), (100, 200), (22, 149, 242), 5, 10, True)

    # 画月亮
    process = cv2.ellipse(process, (400, 300), (200, 200), 0, 315, 405, (65, 214, 254), 2)
    process = cv2.ellipse(process, (541, 300), (141, 141), 0, 270, 450, (65, 214, 254), 2)
    out.write(process)

    draw3 = process.copy()
    for i in range(158, 442):
        p = []
        for j in range(541, 682):
            if(draw3[i,j,0] != 92):
                p.append(j)
        draw3[i, p[0]:p[len(p) - 1], 0] = 65
        draw3[i, p[0]:p[len(p) - 1], 1] = 214
        draw3[i, p[0]:p[len(p) - 1], 2] = 254

    for i in range(71):
        process[158 + 4 * i:158 + 4 * (i + 1), 541:682, 0:3] = draw3[158 + 4 * i:158 + 4 * (i + 1), 541:682, 0:3]
        out.write(process)

    return process

#绘制第二幅简笔画
def drawHouse(out,img):
    process = img
    #屋顶
    process = drawLine(out, process, (400, 100), (200, 300), (5, 57, 95), 3, 10, True)
    process = drawLine(out, process, (400, 100), (600, 300), (5, 57, 95), 3, 10, True)
    process = drawLine(out, process, (200, 300), (600, 300), (5, 57, 95), 3, 10, False)

    roof = process.copy()
    for i in range(200):
        roof[(100+i):(100+i+1),(400-i):(400+i),0] = 5
        roof[(100 + i):(100 + i + 1), (400 - i):(400 + i), 1] = 57
        roof[(100 + i):(100 + i + 1), (400 - i):(400 + i), 2] = 95

    for i in range(25):
        process[(100+8*i):(100+8*(i+1)),200:600,:] = roof[(100+8*i):(100+8*(i+1)),200:600,:]
        out.write(process)
    #房身
    process = drawLine(out, process, (200, 300), (200, 500), (5, 57, 95), 3, 10, True)
    process = drawLine(out, process, (600, 300), (600, 500), (5, 57, 95), 3, 10, True)
    process = drawLine(out, process, (200, 500), (600, 500), (5, 57, 95), 3, 10, False)

    house = process.copy()
    house[300:500,200:600,0] = 121
    house[300:500, 200:600, 1] = 224
    house[300:500, 200:600, 2] = 242

    for i in range(50):
        process[(303+4*i):(303+4*(i+1)),203:598,:] = house[(303+4*i):(303+4*(i+1)),203:598,:]
        out.write(process)

    #窗户
    process = drawLine(out, process, (250, 350), (250, 400), (5, 57, 95), 3, 5, True)
    process = drawLine(out, process, (250, 350), (300, 350), (5, 57, 95), 3, 5, False)
    process = drawLine(out, process, (300, 350), (300, 400), (5, 57, 95), 3, 5, True)
    process = drawLine(out, process, (250, 400), (300, 400), (5, 57, 95), 3, 5, False)

    window = process.copy()
    window[350:400, 250:300, 0] = 39
    window[350:400, 250:300, 1] = 161
    window[350:400, 250:300, 2] = 229

    for i in range(25):
        process[(353+2*i):(353+2*(i+1)),253:298,:] = window[(353+2*i):(353+2*(i+1)),253:298,:]
        out.write(process)

    #门
    process = drawLine(out, process, (475, 425), (475, 505), (5, 57, 95), 3, 10, True)
    process = drawLine(out, process, (475, 425), (525, 425), (5, 57, 95), 3, 10, False)
    process = drawLine(out, process, (525, 425), (525, 505), (5, 57, 95), 3, 10, True)

    door = process.copy()
    door[425:500, 475:525, 0] = 204
    door[425:500, 475:525, 1] = 220
    door[425:500, 475:525, 2] = 44

    for i in range(25):
        process[(428+3*i):(428+3*(i+1)),478:523,:] = door[(428+3*i):(428+3*(i+1)),478:523,:]
        out.write(process)

    return process

#绘制片尾
def drawEnd(out,img,begin,daynight,house,end):
    out.write(begin)
    change1 = img.copy()
    for i in range(25):
        change1 = img.copy()
        image1 = cv2.resize(begin, (775 - 16 * i, 575 - 12 * i))
        change1[0:575 - 12 * i, 0:775 - 16 * i, :] = image1
        out.write(change1)

    out.write(daynight)
    for i in range(25):
        change2 = change1.copy()
        image2 = cv2.resize(daynight, (775 - 16 * i, 575 - 12 * i))
        change2[0:575 - 12 * i, 25 + 16 * i:800, :] = image2
        out.write(change2)

    out.write(house)
    for i in range(25):
        change3 = change2.copy()
        image3 = cv2.resize(house, (775 - 16 * i, 575 - 12 * i))
        change3[25 + 12 * i:600, 0:775 - 16 * i, :] = image3
        out.write(change3)

    out.write(end)
    for i in range(25):
        change4 = change3.copy()
        image4 = cv2.resize(end, (775 - 16 * i, 575 - 12 * i))
        change4[25 + 12 * i:600, 25 + 16 * i:800, :] = image4
        out.write(change4)

    for i in range(40):
        change4[0:600, 20 * i:20 * (i + 1), 0:3] = end[0:600, 20 * i:20 * (i + 1), 0:3]
        out.write(change4)

    for i in range(66):
        change4 = cv2.blur(change4, (1, i + 1))
        out.write(change4)
#写视频
def write():

    #定义视频参数
    fps = 25
    video_name = 'video.avi'
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), fps,(800, 600))
    new_image = np.zeros([600, 800, 3], np.uint8) + 255

    #绘制开头
    process = drawBegin(out,new_image)
    begin = process.copy()

    #绘制第一次过渡
    change1 = process
    for i in range(20):
        change1 += np.uint8((255 - process) / 4)
        out.write(change1)

    #绘制第一幅简笔画
    process = drawDayNight(out,new_image)
    daynight = process.copy()

    #绘制第二次过渡
    original = process.copy()
    for j in range(50):
        temp4 = original[:,8*(j+1):400,:]
        process[:,0:400-8*(j+1),:] = temp4
        temp5 = original[:, 400:800-8*(j+1), :]
        process[:, 400 + 8 * (j + 1):800, :] = temp5
        process[:,(400-8*j):(400+8*j),:] = 255
        out.write(process)
    process[:,0:8,:] = 255
    process[:,792:800,:] = 255
    out.write((process))

    #绘制第二幅简笔画
    process = drawHouse(out,process)
    house = process.copy()

    #第三次过渡
    image2 = process[100:600, 180:620, :].copy()
    for i in range(8):
        image2 = cv2.resize(image2, (400 - 50 * i, 400 - 50 * i))
        photo = new_image.copy()
        photo[(100 + 25 * i):(500 - 25 * i), (200 + 25 * i):(600 - 25 * i), 0:3] = image2
        out.write(photo)
    out.write(new_image)

    #绘制最后一帧画面
    end = new_image.copy()
    cv2.putText(end, 'See you next time :D ', (225, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
    cv2.putText(end, '317010xxxx, Haha Hahah ', (200, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

    #制作片尾
    drawEnd(out,new_image,begin,daynight,house,end)


    cv2.destroyAllWindows()
    return video_name

#读视频
def read(video_name):
    cap = cv2.VideoCapture('video.avi')
    stop = False
    while (cap.isOpened()):
        ret, frame = cap.read()
        if(ret):
            k = cv2.waitKey(40)
            cv2.imshow('video', frame)

            #按空格暂停和继续
            if(k & 0xff == ord(' ')):
                cv2.waitKey(0)

            #按q退出
            if (k & 0xff == ord('q')):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    video_name = write()
    read(video_name)

if __name__=="__main__":
    main()