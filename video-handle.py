import cv2
import numpy as np
import time

# 相似度
global_same_degree = 1
# Otsu阈值
global_otsu_threshold = 131
# 矩形判断
global_rect_xy = []
# 取视频帧间隔
global_frame_interval = 1
video_src = "D:/pycharm-professional-2019.1.1/workspace/demo/source/22.mp4"


def video_handle(cap):
    ticks = time.time()
    framenum = int(cap.get(7))
    framerate = int(cap.get(5))
    print("帧率：", framerate, "帧数：", framenum)
    # 存储经过预处理的图片
    image_deal = []
    # 存储每张图片标准差
    devs = []
    image_pre = None
    for i in range(framenum):
        flag, image = cap.read()
        # 视频帧间隔
        if flag and i % global_frame_interval == 0:
            # 预处理将图片转换为灰度图
            image_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
            image_deal.append(image)
            # 标准差计算 cv2计算效率相比np快两倍
            rgb_values, dev = cv2.meanStdDev(image_gray)
            # dev = np.std(image_gray)
            devs.append(int(dev))
            # cv2.imwrite("D:/pycharm-professional-2019.1.1/workspace/demo/target/tmp/" + str(i) + "dev" + str(int(dev[0])) + ".jpg",
            #             image)
    print("video_handle-第一个for耗时：", time.time()-ticks)
    # 方差不稳定的索引位置
    devs_changes_index = 0
    # 存储方差稳定的索引范围
    devs_unchanges = []
    # 方差稳定的判断范围
    image_range = framerate/3
    # 经过预处理后的图片帧数
    framenum = image_deal.__len__()

    i = 0
    min_range = int(image_range)
    while True:
        if devs_changes_index+min_range+i >= devs.__len__():
            rgb_values, dev = cv2.meanStdDev(np.array(devs[devs_changes_index: devs.__len__()]))
            if not dev > 0.5:
                devs_unchanges.append([devs_changes_index, devs_changes_index+min_range+i-1])
            break
        rgb_values, dev = cv2.meanStdDev(np.array(devs[devs_changes_index: devs_changes_index+min_range+i]))
        if dev > 0.5:
            if i > 0:
                devs_unchanges.append([devs_changes_index, devs_changes_index+min_range+i-1])
            devs_changes_index = devs_changes_index+min_range+i
            i = 0
            continue
        i += 1

    # 存储最终处理的图片
    image_final = []
    for devs_unchange in devs_unchanges:
        # 取方差稳定范围内方差值最大的图片
        dev_max_index = devs_unchange[0] + int(image_range / 3)
        for z in range(devs_unchange[0] + int(image_range / 3), devs_unchange[1] - int(image_range/3)):
            if devs[z] >= devs[dev_max_index]:
                dev_max_index = z
        image_final.append(dev_max_index)
    for image_index in image_final:
        image = image_deal[image_index]
        x, y, w, h = paper_rect_handle(image)
        shape = image.shape
        # 获取纸张区域面积小于总图片面积50%，默认为失效图片
        if (w * h) / (shape[0] * shape[1]) >= 0.5:
            # 裁剪坐标为[y0:y1, x0:x1]
            cropped = image[y:y + h, x:x + w]
            # 检测纸张区域是否被遮盖
            if cr_otsu(cropped) < global_otsu_threshold:
                # 检测图片相似度并去重
                if image_pre is None or same_imgcolor_handle(image, image_pre) <= global_same_degree:
                    image_pre = image
                    cv2.imwrite("D:/pycharm-professional-2019.1.1/workspace/demo/target/" + str(image_index) + ".jpg",
                                image)
    cap.release()
    print("video_handle耗时：", time.time()-ticks)


# 获取纸张区域
def paper_rect_handle(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 对cr通道分量进行高斯滤波
    S_GB = cv2.GaussianBlur(hsv[..., 1], (5, 5), 0)
    # 根据OTSU算法求图像阈值, 对图像进行二值化
    # 白色为255，黑色为0
    # Threshold Binary：将大于阈值的灰度值设为最大灰度值，小于阈值的值设为0。
    rec, paper = cv2.threshold(S_GB, 20, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rect = []
    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        rect.append([x, y, w, h])
        # cv2.rectangle(img, (x, y), (x + w, y + h), (153, 153, 0), 5)
        # rect = cv2.minAreaRect(contours[i])
        # box = np.int0(cv2.boxPoints(rect))  # 通过box会出矩形框
    rect.sort(key=lambda r: (r[2] * r[3]), reverse=True)
    # cv2.rectangle(img, (rect[0][0], rect[0][1]), (rect[0][0] + rect[0][2], rect[0][1] + rect[0][3]), (153, 153, 0), 5)
    # cv2.rectangle(img, (x, y), (x + w, y + h), (153, 153, 0), 5)
    # cv2.imwrite("D:/pycharm-professional-2019.1.1/workspace/demo/plt/27.mp4/" + str(b) + ".jpg", img)
    return rect[0][0], rect[0][1], rect[0][2], rect[0][3]


# YCrCb颜色空间的Cr分量+Otsu阈值分割
def cr_otsu(img):
    # 肤色检测: YCrCb之Cr分量 + OTSU二值化
    # 把图像转换到YUV色域
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # 图像分割, 分别获取y, cr, br通道图像
    (y, cr, cb) = cv2.split(ycrcb)
    # 高斯滤波, cr 是待滤波的源图像数据, (5,5)是值窗口大小, 0 是指根据窗口大小来计算高斯函数标准差
    # 对cr通道分量进行高斯滤波
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
    # 根据OTSU算法求图像阈值, 对图像进行二值化
    rec, skin1 = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(rec)


# 计算单通道的直方图的相似值
def calculate(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree


# 第二种对比两张图片相似度-彩色图rgb三通道
def same_imgcolor_handle(image1, image2, size=(256, 256)):
    # RGB每个通道的直方图相似度
    # 将图像resize后，分离为RGB三个通道，再计算每个通道的相似值
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += calculate(im1, im2)
    sub_data = sub_data / 3
    return sub_data


# 过滤缺陷图片：通过矩形判断
def scharr_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rect = []
    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        rect.append([x, y, w, h])
    rect.sort(key=lambda r: (r[2]*r[3]), reverse=True)
    global global_rect_xy
    flag = True
    if global_rect_xy.__len__() == 0:
        global_rect_xy = [rect[1][2], rect[1][3]]
    if abs(rect[1][2]-global_rect_xy[0])/global_rect_xy[0] > 0.2 or abs(rect[1][3]-global_rect_xy[1])/global_rect_xy[1] > 0.2:
        flag = False
    return flag


if __name__ == "__main__":
    cap = cv2.VideoCapture(video_src)
    video_handle(cap)