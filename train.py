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
    def __init__(self, file_path='./test2.jpg', src=None, DEBUG=False):
        self.DEBUG = DEBUG

        self.CARD_WIDTH = 200
        self.CARD_HEIGHT = 304  # int(self.CARD_WIDTH * 1.52777778)

        self.INFO_WIDTH = int((7/30) * self.CARD_WIDTH)
        self.INFO_HEIGHT = int((7/18) * self.CARD_HEIGHT)

        self.card_container = np.array(
            [[0, 0], [self.CARD_WIDTH, 0], [self.CARD_WIDTH, self.CARD_HEIGHT], [0, self.CARD_HEIGHT]], np.float32)

        self.approx = None
        if src is not None:
            self.image = src
        else:
            self.image = cv2.imread(file_path)
        self.og_image = self.image.copy()

    def filter_boundingRect(self, c):
        x, y, w, h = cv2.boundingRect(c)
        is_big_enough = (w*h) > (0.1*self.INFO_WIDTH * self.INFO_HEIGHT)
        not_too_big = (w*h) < (0.5*self.INFO_WIDTH * self.INFO_HEIGHT)
        rectangular_like = (w/h) < 1.5 and (w/h) > 0.5
        return is_big_enough and not_too_big and rectangular_like

    def image_is_blurry(self, image):
        return cv2.Laplacian(image, cv2.CV_64F).var() < 100

    def debug_print(self, *text):
        if self.DEBUG:
            print(text)

    def preprocess_image(self):
        self.IMAGE_HEIGHT, self.IMAGE_WIDTH = self.image.shape[:2]

        self.gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # check if image is blurry
        if self.image_is_blurry(self.gray_img):
            raise Exception("Image is blurry")

        _, self.gray_img = cv2.threshold(
            self.gray_img, 100, 255, cv2.THRESH_BINARY)

        kernel = np.ones((30, 30), np.uint8)
        closing = cv2.morphologyEx(self.gray_img, cv2.MORPH_CLOSE, kernel)

        edges = cv2.Canny(closing, 100, 200)

        kernel = np.ones((3, 3), np.uint8)
        self.edges = cv2.dilate(edges, kernel, iterations=1)

    def get_corners(self):
        cnts, _ = cv2.findContours(
            self.edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        card = cnts[0]

        # Approximate the corner points of the card
        peri = cv2.arcLength(card, True)
        self.approx = cv2.approxPolyDP(card, 0.01*peri, True)
        x, y, w, h = cv2.boundingRect(card)
        self.poi = np.float32(self.approx.reshape(4, 2))

        self.poi = np.asarray(list(self.poi), np.float32)
        self.poi = np.asarray(
            sorted(np.concatenate([self.poi]).tolist()), np.float32)

        self.debug_print("pois: ", self.poi)

        self.is_horizontal(x, y, w, h)

    def is_horizontal(self, x, y, w, h):
        if w >= 1.3*h:  # If card is horizontally oriented
            self.debug_print("horizontal!!!")
            self.image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
            self.og_image = cv2.rotate(
                self.og_image, cv2.ROTATE_90_CLOCKWISE)
            self.preprocess_image()
            self.get_corners()

            temp = self.poi[0].copy()
            if self.poi[0][1] < self.poi[3][1]:
                self.poi[0] = self.poi[3]
                self.poi[3] = temp
                self.debug_print("swapping POI: ", self.poi)

    def final_process(self):
        H, _ = cv2.findHomography(self.poi[[0, 2, 3, 1]], self.card_container, method=cv2.RANSAC,
                                  ransacReprojThreshold=3.0)

        M = cv2.getPerspectiveTransform(
            self.poi[[0, 2, 3, 1]], self.card_container)

        # Sharpen Original Image And Crop
        original_image = cv2.warpPerspective(
            self.image, H, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT), flags=cv2.INTER_LINEAR)[0:self.CARD_HEIGHT, 0:self.CARD_WIDTH]

        gray_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        _, gray_img = cv2.threshold(gray_img, 170, 255, cv2.THRESH_BINARY)

        possible_corners = [gray_img[0:self.INFO_HEIGHT, 0:self.INFO_WIDTH],
                            cv2.flip(gray_img[0:self.INFO_HEIGHT, self.CARD_WIDTH-self.INFO_WIDTH:self.CARD_WIDTH], 1)]
        original_images = [original_image[0:self.INFO_HEIGHT, 0:self.INFO_WIDTH],
                           cv2.flip(original_image[0:self.INFO_HEIGHT, self.CARD_WIDTH-self.INFO_WIDTH:self.CARD_WIDTH], 1)]

        self.final_cnts = None
        for i, possible_corner in enumerate(possible_corners):
            self.info_gray = 255 - possible_corner
            self.info_colored = original_images[i]
            cnts = cv2.findContours(self.info_gray, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]
            cnts = list(filter(self.filter_boundingRect, cnts))
            cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[1])

            if len(cnts) == 2:
                self.final_cnts = cnts
                break

        if self.final_cnts is None:
            raise Exception("Can't find any rank and suit")

        for c in self.final_cnts:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(self.info_colored, (x, y),
                          (x+w, y+h), (36, 255, 12), 2)

    def get_bounding_box_image(self, box_index):
        x, y, w, h = cv2.boundingRect(self.final_cnts[box_index])
        return self.info_gray[y:y+h, x:x+w]

    def save_bounding_box(self, box_index, filename):
        x, y, w, h = cv2.boundingRect(self.final_cnts[box_index])
        cv2.imwrite(filename, self.info_gray[y:y+h, x:x+w])

    def run(self):
        try:
            self.preprocess_image()
            self.get_corners()
            self.final_process()
        except Exception as e:
            self.debug_print("[!]", e)
            return False

        if self.DEBUG:
            self.show_debug()

        cv2.imwrite("output2.jpg", self.info_colored)
        return True

    def show_debug(self, wait=True):
        # for index, pt in enumerate(self.poi):
        #     x, y = pt
        #     cv2.circle(original_image, (int(x), int(y)), radius=5,
        #                color=(255, 0, 0), thickness=5)

        #     character = chr(65 + index)
        #     cv2.putText(original_image, character, (int(x), int(y)),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        # draw everything
        cv2.imshow("Image - Original", self.image)
        cv2.imshow("Image - Edges", self.edges)
        cv2.imshow("Image - Info Gray", self.info_gray)
        cv2.imshow("Image - Info", self.info_colored)

        if wait:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()


if __name__ == "__main__":
    trainer = CardScanner("./test.jpg", DEBUG=True)

    trainer.run()
    trainer.save_bounding_box(0, "./test4_rank.jpg")
    trainer.save_bounding_box(1, "./test4_suit.jpg")
