import cv2
import canny

image = cv2.imread("C:\\Users\\26317\\PycharmProjects\\canny\\Pic1_1.bmp")
cv2.imshow("src", image)
result = canny.Canny(image, 85, 95)
cv2.imshow("dst", result)
cv2.waitKey(0)
