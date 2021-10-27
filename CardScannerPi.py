import cv2
import numpy as np
import os
import sys
import json
import platform
from operator import itemgetter
from numpy.lib.function_base import angle

from picamera.array import PiRGBArray
from picamera import PiCamera


class CardScannerPi:
    def __init__(self, file_path='./test2.jpg', config_path='./config.json', src=None, DEBUG=False):
        if not os.path.exists(config_path):
            print("Config File Not Found. Please run ./configurator.py")
            sys.exit(1)

        config = json.load(open(config_path))

        self.DEBUG = DEBUG

        self.INFO_WIDTH = config.get('x_width')
        self.INFO_HEIGHT = config.get('y_width')
        self.INFO_XOFFSET = config.get('x_offset')
        self.INFO_YOFFSET = config.get('y_offset')

        print(self.INFO_WIDTH, self.INFO_HEIGHT,
              self.INFO_XOFFSET, self.INFO_YOFFSET)

        self.approx = None
        if src is not None:
            self.image = src
        else:
            self.image = cv2.imread(file_path)

        self.image = self.image[self.INFO_YOFFSET:self.INFO_YOFFSET +
                                self.INFO_HEIGHT, self.INFO_XOFFSET:self.INFO_XOFFSET+self.INFO_WIDTH]
        self.og_image = self.image.copy()

    def filter_boundingRect(self, c):
        x, y, w, h = cv2.boundingRect(c)
        is_big_enough = (w*h) > (0.1*self.INFO_WIDTH * self.INFO_HEIGHT)
        not_too_big = (w*h) < (0.5*self.INFO_WIDTH * self.INFO_HEIGHT)
        rectangular_like = (w/h) < 1.5 and (w/h) > 0.5
        return is_big_enough and not_too_big and rectangular_like

    def image_is_blurry(self, image):
        return cv2.Laplacian(image, cv2.CV_64F).var() < 1

    def debug_print(self, *text):
        if self.DEBUG:
            print(text)

    def preprocess_image(self):
        self.IMAGE_HEIGHT, self.IMAGE_WIDTH = self.image.shape[:2]

        self.gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # check if image is blurry
        if self.image_is_blurry(self.gray_img):
            raise Exception("Image is blurry")

        blur = cv2.GaussianBlur(self.image, (13, 13), 0)
        self.gray_img = cv2.adaptiveThreshold(self.gray_img, 255,
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 11)

        contours, _ = cv2.findContours(
            self.gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=cv2.contourArea,
                          reverse=True)[:2]  # sort by area
        self.final_cnts = sorted(
            contours, key=lambda c: cv2.boundingRect(c)[1])  # sort by y

    def get_bounding_box_image(self, box_index):
        x, y, w, h = cv2.boundingRect(self.final_cnts[box_index])
        return self.gray_img[y:y+h, x:x+w]

    def save_bounding_box(self, box_index, filename):
        x, y, w, h = cv2.boundingRect(self.final_cnts[box_index])
        cv2.imwrite(filename, self.gray_img[y:y+h, x:x+w])

    def run(self):
        try:
            self.preprocess_image()
        except Exception as e:
            self.debug_print("[!]", e)
            return False

        if self.DEBUG:
            self.show_debug()

        return True

    def show_debug(self, wait=True):
        self.debug_gray = self.gray_img.copy()
        self.debug_img = self.image.copy()
        for c in self.final_cnts:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(self.image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.drawContours(self.debug_img, self.final_cnts, -1, (0, 255, 0), 3)
        cv2.imshow("Image - Original", self.image)
        cv2.imshow("Image - Gray", self.debug_gray)

        if wait:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()


if __name__ == "__main__":
    cv2.setUseOptimized(True)

    # define a video capture object
    font = cv2.FONT_HERSHEY_SIMPLEX

    file_names = ['Ace', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight',
                  'Nine', 'Ten', 'Jack', 'Queen', 'King', 'Spades', 'Diamonds',
                  'Clubs', 'Hearts']
    file_index = 0
    file_path = os.path.dirname(os.path.abspath(__file__)) + '/cards/'
    print(file_path)

    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 10
    rawCapture = PiRGBArray(camera, size=(640, 480))

    while file_index < len(file_names):
        file_name = f'{file_names[file_index]}.jpg'

        # Capture the video frame
        # by frame
        print(f"[{file_name}] Press 'p' to take picture', q to quit")
        saved_info_gray = None
        saved_final_cnts = None
        for image in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            frame = image.array

            scanner = CardScannerPi(src=frame, DEBUG=False)
            res = scanner.run()

            if res:
                saved_info_gray = scanner.gray_img
                saved_final_cnts = scanner.final_cnts

            cv2.imshow('frame', frame)
            rawCapture.truncate(0)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                camera.close()
                cv2.destroyAllWindows()
                sys.exit(0)
            if key == ord('p'):
                bb = 0 if file_index < 13 else 1
                fp = file_path + file_name
                if (saved_info_gray is not None) and (saved_final_cnts is not None):
                    x, y, w, h = cv2.boundingRect(saved_final_cnts[bb])
                    cv2.imwrite(fp, saved_info_gray[y:y+h, x:x+w])
                    break

        file_index += 1

    # After the loop release the cap object
    camera.close()
    # Destroy all the windows
    cv2.destroyAllWindows()
