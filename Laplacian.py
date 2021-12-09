import numpy as np
import cv2

image = cv2.imread("D:\\Work-Coding\\_Projects\\PycharmProject\\ACannyAchieve\\Pic1_1.bmp")
# 将图像转化为灰度图像
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", image)
# 拉普拉斯边缘检测
lap = cv2.Laplacian(image, cv2.CV_64F)
# 对lap去绝对值
lap = np.uint8(np.absolute(lap))
cv2.imshow("Laplacian", lap)
cv2.waitKey()
