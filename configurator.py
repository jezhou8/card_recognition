import cv2
import numpy as np
import platform
import sys


cv2.setUseOptimized(True)

# define a video capture object
if platform.system() == 'Windows':
    vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
else:
    vid = cv2.VideoCapture(0)

print("Press P to take a Picture, Q to quit")

window_name = "Cam View"
cv2.namedWindow(winname=window_name)
cv2.setMouseCallback("Title of Popup Window", draw_circle)
while True:
    ret, frame = vid.read()

    cv2.imshow(frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):
        pass
    if key == ord('q'):
        break

vid.release()
cv2.destoryAllWindows()
sys.exit(0)


def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("hello")
        cv2.circle(img, (x, y), 100, (0, 255, 0), -1)


while True:
    cv2.imshow("Title of Popup Window", img)
    if cv2.waitKey(10) & 0xFF == 27:
        break
cv2.destroyAllWindows()
