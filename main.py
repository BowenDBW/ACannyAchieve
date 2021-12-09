import cv2
import cannyByDBW

image = cv2.imread("D:\\Work-Coding\\_Projects\\PycharmProject\\ACannyAchieve\\Pic1_1.bmp")
cv2.imshow("src", image)
result = cannyByDBW.Canny(image, 85, 95)
cv2.imshow("dst", result)
cv2.waitKey(0)
