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


class CardScanner:
    def __init__(self, file_path='./test2.jpg', src=None):
        self.CARD_WIDTH = 200
        self.CARD_HEIGHT = int(self.CARD_WIDTH * 1.52777778)

        self.INFO_WIDTH = int((7/30) * self.CARD_WIDTH)
        self.INFO_HEIGHT = int((7/18) * self.CARD_HEIGHT)

        self.card_container = np.array(
            [[0, 0], [self.CARD_WIDTH, 0], [self.CARD_WIDTH, self.CARD_HEIGHT], [0, self.CARD_HEIGHT]], np.float32)

        if src is not None:
            self.image = src
        else:
            self.image = cv2.imread(file_path)
        self.IMAGE_HEIGHT, self.IMAGE_WIDTH = self.image.shape[:2]

    def line_eq(self, p1, p2):
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0]*p2[1] - p2[0]*p1[1])
        return A, B, -C

    def intersection(self, L1, L2):
        D = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]
        if D != 0:
            x = Dx / D
            y = Dy / D
            return x, y
        else:
            return False


    def angleBetween2Lines(self, line1, line2):
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


    def is_point_close(self, pt, arr):
        x1, y1 = pt
        for pt1 in arr:
            x2, y2 = pt1
            dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
            if dist < 20:
                return True

        return False


    def filter_boundingRect(self, c):
        x, y, w, h = cv2.boundingRect(c)
        return (w*h) > (0.1*self.INFO_WIDTH * self.INFO_HEIGHT)


    def preprocess_image(self):
        self.image = cv2.fastNlMeansDenoisingColored(self.image, None, 10, 10, 7, 21)
        self.original_image = self.image.copy()

        gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gray_img[gray_img > 150] = 255
        gray_img[gray_img <= 150] = 0
        self.gray_img = gray_img.copy()

        kernel = np.ones((30, 30), np.uint8)
        closing = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel)
        edges = cv2.Canny(closing, 100, 200)

        kernel = np.ones((3, 3), np.uint8)
        self.edges = cv2.dilate(edges, kernel, iterations=1)

    def get_lines(self):
        rho = 2  # distance resolution in pixels of the Hough grid
        theta = np.pi / 720  # angular resolution in radians of the Hough grid
        threshold = 100  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 100  # minimum number of pixels making up a line
        max_line_gap = 5  # maximum gap in pixels between connectable line segments

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        self.lines = cv2.HoughLinesP(self.edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)

        for line in self.lines:
            for x1, y1, x2, y2 in line:
                cv2.line(self.image, (x1, y1), (x2, y2), (random.randint(50, 240),
                                                    random.randint(50, 240), random.randint(50, 240)), 5)
        

    def get_corners(self):
        poi = set()
        for line1 in self.lines:
            for line2 in self.lines:
                if np.array_equal(line1, line2):
                    continue

                x1, y1, x2, y2 = line1[0]
                x3, y3, x4, y4 = line2[0]

                l1 = [(x1, y1), (x2, y2)]
                l2 = [(x3, y3), (x4, y4)]

                angle_diff = self.angleBetween2Lines(l1, l2)
                angle_threshold = 0.9
                if angle_diff > angle_threshold:
                    continue

                eq1 = self.line_eq((x1, y1), (x2, y2))
                eq2 = self.line_eq((x3, y3), (x4, y4))

                intersect = self.intersection(eq1, eq2)
                if (intersect):
                    x, y = intersect
                    if (x < 0) or x > self.IMAGE_WIDTH or y < 0 or y > self.IMAGE_HEIGHT or self.is_point_close(intersect, list(poi)):
                        continue
                    poi.add((int(x), int(y)))
        
        if len(poi) < 4:
            raise Exception("Not enough corners found")
            
        self.poi = np.asarray(list(poi), np.float32)
        self.poi = np.asarray(sorted(np.concatenate([self.poi]).tolist()), np.float32)        
                      
        print("pois: ", poi)
# dilate edges

    def final_process(self):
        H, _ = cv2.findHomography(self.poi[[0, 2, 3, 1]], self.card_container, method=cv2.RANSAC,
                                ransacReprojThreshold=3.0)

        # Sharpen Original Image And Crop
        original_image = cv2.warpPerspective(
            self.original_image, H, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT), flags=cv2.INTER_LINEAR)[0:self.CARD_HEIGHT, 0:self.CARD_WIDTH]

        gray_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        gray_img[gray_img > 170] = 255
        gray_img[gray_img <= 170] = 0


        possible_corners = [gray_img[0:self.INFO_HEIGHT, 0:self.INFO_WIDTH],
                            cv2.flip(gray_img[0:self.INFO_HEIGHT, self.CARD_WIDTH-self.INFO_WIDTH:self.CARD_WIDTH], 1)]
        original_images = [original_image[0:self.INFO_HEIGHT, 0:self.INFO_WIDTH],
                            cv2.flip(original_image[0:self.INFO_HEIGHT, self.CARD_WIDTH-self.INFO_WIDTH:self.CARD_WIDTH], 1)]

        self.final_cnts = None
        for i, possible_corner in enumerate(possible_corners):
            info_gray = 255 - possible_corner
            self.info_colored = original_images[i]
            cnts = cv2.findContours(info_gray, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]
            cnts = list(filter(self.filter_boundingRect, cnts))
            cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[1])[:2]

            if len(cnts) == 2:
                self.final_cnts = cnts
                break
        
        if self.final_cnts is None:
            raise Exception("Can't find any rank and suit")

        for c in self.final_cnts:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(self.info_colored, (x, y), (x+w, y+h), (36, 255, 12), 2)


    def run(self):
        try:
            self.preprocess_image()
            self.get_lines()
            self.get_corners()
            self.final_process()
        except Exception as e:
            print(e)
            return False

        cv2.imwrite("output2.jpg", self.info_colored)
        return True
        


if __name__ == "__main__":
    trainer = CardScanner("./test.jpg")
    start = time.time()
    trainer.run()
    end = time.time()
    print(end - start)