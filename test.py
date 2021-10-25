import numpy as np
import cv2

image = cv2.imread('./test4.jpg')
image = cv2.rotate(image, -1)
cv2.imwrite("./test5.jpg", image)
