import cv2
import numpy as np
import time
import os
import random
import sys
import math
from operator import itemgetter

from numpy.lib.function_base import angle

DEBUG = False


IM_WIDTH = 640
IM_HEIGHT = 480

RANK_WIDTH = 70
RANK_HEIGHT = 125

SUIT_WIDTH = 70
SUIT_HEIGHT = 100

CARD_WIDTH = 400
CARD_HEIGHT = int(CARD_WIDTH * 1.52777778)

INFO_WIDTH = int((7/30) * CARD_WIDTH)
INFO_HEIGHT = int((7/18) * CARD_HEIGHT)
print(INFO_WIDTH, INFO_HEIGHT)
card_container = np.array(
    [[0, 0], [CARD_WIDTH-1, 0], [CARD_WIDTH-1, CARD_HEIGHT-1], [0, CARD_HEIGHT-1]], np.float32)

image = cv2.imread("./test5.jpg")


def flattener(image, pts, w, h):
    """Flattens an image of a card into a top-down 200x300 perspective.
    Returns the flattened, re-sized, grayed image.
    See www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/"""
    temp_rect = np.zeros((4, 2), dtype="float32")

    print('pts', pts)
    s = np.sum(pts, axis=2)
    print('s', s)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis=-1)
    print('diff', diff)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    print(tl, tr, br, bl)
    # Need to create an array listing points in order of
    # [top left, top right, bottom right, bottom left]
    # before doing the perspective transform

    if w <= 0.8*h:  # If card is vertically oriented
        print("vert!!!!!!")
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2*h:  # If card is horizontally oriented
        print("horizontal!!!")
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow("Image Rotated", image)
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    # If the card is 'diamond' oriented, a different algorithm
    # has to be used to identify which point is top left, top right
    # bottom left, and bottom right.

    if w > 0.8*h and w < 1.2*h:  # If card is diamond oriented
        # If furthest left point is higher than furthest right point,
        # card is tilted to the left.
        if pts[1][0][1] <= pts[3][0][1]:
            # If card is titled to the left, approxPolyDP returns points
            # in this order: top right, top left, bottom left, bottom right
            temp_rect[0] = pts[1][0]  # Top left
            temp_rect[1] = pts[0][0]  # Top right
            temp_rect[2] = pts[3][0]  # Bottom right
            temp_rect[3] = pts[2][0]  # Bottom left

        # If furthest left point is lower than furthest right point,
        # card is tilted to the right
        elif pts[1][0][1] > pts[3][0][1]:
            # If card is titled to the right, approxPolyDP returns points
            # in this order: top left, bottom left, bottom right, top right
            temp_rect[0] = pts[0][0]  # Top left
            temp_rect[1] = pts[3][0]  # Top right
            temp_rect[2] = pts[2][0]  # Bottom right
            temp_rect[3] = pts[1][0]  # Bottom left

    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    maxWidth = 200
    maxHeight = 300

    print("temp_rect: ", temp_rect)
    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    dst = np.array([[0, 0], [maxWidth-1, 0], [maxWidth-1,
                   maxHeight-1], [0, maxHeight-1]], np.float32)

    H, _ = cv2.findHomography(temp_rect, dst, method=cv2.RANSAC,
                              ransacReprojThreshold=3.0)
    M = cv2.getPerspectiveTransform(temp_rect, dst)
    warp = cv2.warpPerspective(image, H, (width, height))
    warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)

    return warp


# Pre-process image
print(image.shape)
height, width = image.shape[:2]
image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
original_image = image.copy()

gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_img[gray_img > 150] = 255
gray_img[gray_img <= 150] = 0


kernel = np.ones((30, 30), np.uint8)
closing = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel)
edges = cv2.Canny(closing, 100, 200)

# dilate edges
kernel = np.ones((2, 2), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=1)

cnts, hier = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
card = cnts[0]

# Approximate the corner points of the card
peri = cv2.arcLength(card, True)
print(peri)
approx = cv2.approxPolyDP(card, 0.01*peri, True)
x, y, w, h = cv2.boundingRect(card)
cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
pts = np.float32(approx)
cv2.drawContours(image, approx, -1, (0, 255, 0), 5)
# Flatten the card and convert it to 200x300
warp = flattener(image, pts, w, h)


cv2.imshow("Image - Original", image)
cv2.imshow("Image - Edges", edges)
cv2.imshow("Image - Warped", warp)

key = cv2.waitKey(0) & 0xFF
if key == ord('q'):
    cv2.destroyAllWindows()
