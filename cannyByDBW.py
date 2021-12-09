import cv2
import numpy as np
import math


def Canny(img, threshold_min, threshold_max):
    # 高斯滤波
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # new_gray = cv2.GaussianBlur(gray, (5, 5), 1)
    new_gray = gray

    gaussian_result = np.copy(new_gray)
    cv2.imshow("gaussian", gaussian_result)

    # 梯度幅值,用 sobel 算子
    W1, H1 = new_gray.shape[:2]
    dx = np.zeros([W1 - 1, H1 - 1])
    dy = np.zeros([W1 - 1, H1 - 1])
    d = np.zeros([W1 - 1, H1 - 1])
    dDegree = np.zeros([W1 - 1, H1 - 1])
    for i in range(1, W1 - 1):
        # thread
        for j in range(1, H1 - 1):
            dx[i, j] = new_gray[i - 1, j - 1] + 2 * new_gray[i, j - 1] + \
                       new_gray[i + 1, j - 1] - new_gray[i - 1, j + 1] - \
                       2 * new_gray[i, j + 1] - new_gray[i + 1, j + 1]

            dy[i, j] = new_gray[i - 1, j - 1] + 2 * new_gray[i - 1, j] + new_gray[i - 1, j + 1] - new_gray[i + 1, j - 1] \
                       - 2 * new_gray[i + 1, j] - new_gray[i + 1, j + 1]
            d[i, j] = np.sqrt(np.square(dx[i, j]) + np.square(dy[i, j]))
            dDegree[i, j] = math.degrees(math.atan2(dy[i, j], dx[i, j]))
            if dDegree[i, j] < 0:
                dDegree += 360

    d_r = np.uint8(np.copy(d))
    cv2.imshow("gradient", d_r)

    # 非极大值抑制
    W2, H2 = d.shape
    NMS = np.copy(d)
    NMS[0, :] = NMS[W2 - 1, :] = NMS[:, 0] = NMS[:, H2 - 1] = 0
    for i in range(1, W2 - 1):
        for j in range(1, H2 - 1):
            if d[i, j] == 0:
                NMS[i, j] = 0
            else:
                if (22.5 >= dDegree[i, j] >= 0) or (dDegree[i, j] >= 337.5):
                    g1 = NMS[i, j - 1]
                    g2 = NMS[i, j + 1]
                elif (67.5 >= dDegree[i, j] > 22.5) or (337.5 >= dDegree[i, j] > 292.5):
                    g1 = NMS[i - 1, j + 1]
                    g2 = NMS[i + 1, j - 1]
                elif (112.5 >= dDegree[i, j] > 67.5) or (292.5 >= dDegree[i, j] > 247.5):
                    g1 = NMS[i - 1, j]
                    g2 = NMS[i + 1, j]
                elif (157.5 >= dDegree[i, j] > 112.5) or (247.5 >= dDegree[i, j] > 202.5):
                    g1 = NMS[i - 1, j - 1]
                    g2 = NMS[i + 1, j + 1]
                else:
                    g1 = NMS[i, j - 1]
                    g2 = NMS[i, j + 1]
                if NMS[i, j] < g1 or NMS[i, j] < g2:
                    NMS[i, j] = 0

    # 双阔值算法检测，连接边缘
    W3, H3 = NMS.shape
    DT = np.zeros([W3, H3])
    # 定义高低阔值
    TL = min(threshold_min, threshold_max)
    TH = max(threshold_min, threshold_max)

    for i in range(1, W3 - 1):
        for j in range(1, H3 - 1):
            if NMS[i, j] < TL:
                DT[i, j] = 0
            elif NMS[i, j] > TH:
                DT[i, j] = 255
            else:
                if NMS[i - 1, j] > TH or NMS[i - 1, j - 1] > TH or NMS[i - 1, j + 1] > TH or NMS[i, j - 1] > TH \
                        or NMS[i, j + 1] > TH or NMS[i + 1, j] > TH or NMS[i + 1, j - 1] > TH or NMS[i + 1, j + 1] > TH:
                    DT[i, j] = 255
    return DT
