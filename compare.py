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

img_path = os.path.dirname(os.path.abspath(__file__)) + '/Card_Imgs/'

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
    [[0, 0], [CARD_WIDTH, 0], [CARD_WIDTH, CARD_HEIGHT], [0, CARD_HEIGHT]], np.float32)


def line_eq(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C


def intersection(L1, L2):
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False


def angleBetween2Lines(line1, line2):
    dx1 = line1[1][0] - line1[0][0]
    dy1 = line1[1][1] - line1[0][1]
    vector1 = [dx1, dy1]

    dx2 = line2[1][0] - line2[0][0]
    dy2 = line2[1][1] - line2[0][1]
    vector2 = [dx2, dy2]

    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector1, unit_vector2)

    return dot_product


def is_point_close(pt, arr):
    x1, y1 = pt
    for pt1 in arr:
        x2, y2 = pt1
        dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
        if dist < 20:
            return True

    return False


# Pre-process image

filenames = ['./test2.jpg', './test3.jpg', './test4.jpg', './test5.jpg']
res_images = []

for filename in filenames:
    image = cv2.imread(filename)

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
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largestContour = contours[0]

    nh, nw = image.shape[:2]
    for cnt in contours:
        x, y, w, h = bbox = cv2.boundingRect(cnt)
        if h >= 0.3 * nh:
            cv2.rectangle(image, (x, y), (x+w, y+h),
                          (0, 255, 0), 1, cv2.LINE_AA)

    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi / 720  # angular resolution in radians of the Hough grid
    # minimum number of votes (intersections in Hough grid cell)
    threshold = 100
    min_line_length = 100  # minimum number of pixels making up a line
    max_line_gap = 5  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    poi = set()
    for line1 in lines:
        for line2 in lines:
            if np.array_equal(line1, line2):
                continue

            x1, y1, x2, y2 = line1[0]
            x3, y3, x4, y4 = line2[0]

            l1 = [(x1, y1), (x2, y2)]
            l2 = [(x3, y3), (x4, y4)]

            angle_diff = angleBetween2Lines(l1, l2)
            angle_threshold = 0.9
            if angle_diff > angle_threshold:
                continue

            eq1 = line_eq((x1, y1), (x2, y2))
            eq2 = line_eq((x3, y3), (x4, y4))

            intersect = intersection(eq1, eq2)
            if (intersect):
                x, y = intersect
                if (x < 0) or x > width or y < 0 or y > height or is_point_close(intersect, list(poi)):
                    continue
                poi.add((int(x), int(y)))

    poi = np.asarray(sorted(np.concatenate([list(poi)]).tolist()), np.float32)

    print("pois: ", poi)

    print("num lines: ", len(lines))

    labeled_pois = {}
    for index, pt in enumerate(poi):
        x, y = pt
        cv2.circle(image, (int(x), int(y)), radius=3,
                   color=(0, 0, 255), thickness=2)

        character = chr(65 + index)
        labeled_pois[character] = (int(x), int(y))
        cv2.putText(image, character, (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # correct labeled points
    corrected_index = [0, 1, 2, 3]
    if (labeled_pois["A"][1] <= labeled_pois["D"][1]):
        # 'left' skewed
        print('left skewed', filename)
        corrected_index = [0, 2, 3, 1]
    elif (labeled_pois["A"][1] > labeled_pois["D"][1]):
        # 'right' skewed
        print('right skewed', filename)
        if (labeled_pois["B"][0] < labeled_pois["C"][0]):
            # minor skew, only need to flip x axis
            print('minor')
            corrected_index = [3, 2, 0, 1]
        else:
            # major skew, needs to flip both axis
            print('major')
            corrected_index = [0, 1, 3, 2]
    else:
        print('couldn"t determine skew')
        sys.exit(1)

    H, _ = cv2.findHomography(poi[corrected_index], card_container, method=cv2.RANSAC,
                              ransacReprojThreshold=3.0)
    homo_image = cv2.warpPerspective(
        image, H, (width, height), flags=cv2.INTER_LINEAR)

    res_images.append(image)

for filename, img in zip(filenames, res_images):
    cv2.imshow(filename, img)

key = cv2.waitKey(0) & 0xFF
if key == ord('q'):
    cv2.imwrite(img_path+"countour.jpg", edges)

cv2.destroyAllWindows()
