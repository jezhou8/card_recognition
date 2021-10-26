import cv2
import time
import os
import sys
import numpy as np
import math
import heapq

from numpy.lib.function_base import diff
from train import DEBUG, CardScanner


cv2.setUseOptimized(True)

# define a video capture object
vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
font = cv2.FONT_HERSHEY_SIMPLEX

file_names = ['Ace', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight',
              'Nine', 'Ten', 'Jack', 'Queen', 'King', 'Spades', 'Diamonds',
              'Clubs', 'Hearts']
file_index = 0
file_path = os.path.dirname(os.path.abspath(__file__)) + '/cards/'

final_height = 75
final_width = 55
rank_training_dict = {}
suit_training_dict = {}


def pad_array(array, final_width, final_height):
    x_pad = (final_width - array.shape[1]) / 2
    x_pad_1 = int(math.floor(x_pad))
    x_pad_2 = int(math.ceil(x_pad))

    y_pad = (final_height - array.shape[0]) / 2
    y_pad_1 = int(math.floor(y_pad))
    y_pad_2 = int(math.ceil(y_pad))

    if x_pad < 0 or y_pad < 0:
        print('Error: padding is negative', array.shape)

    return np.pad(array, ((y_pad_1, y_pad_2), (x_pad_1, x_pad_2)), 'constant')


def get_binary_image(image):
    return np.where(image == 255, 1, image)


def mse(img1, img2):
    err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err /= float(img1.shape[0] * img1.shape[1])

    return err


def load_training_data():
    for i, name in enumerate(file_names):
        fp = file_path + name + '.jpg'
        image = cv2.imread(fp)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bw_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

        bw_array = get_binary_image(bw_image)
        bw_array = pad_array(bw_array, final_width, final_height)

        if i < 13:
            rank_training_dict[name] = bw_array
        else:
            suit_training_dict[name] = bw_array


def classify(image, type='rank', preprocess=True):
    if preprocess:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bw_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        bw_array = get_binary_image(bw_image)
    else:
        bw_array = get_binary_image(image)

    bw_array = pad_array(bw_array, final_width, final_height)

    diff_dict = []
    if type == 'rank':
        for key, value in rank_training_dict.items():
            heapq.heappush(diff_dict, (mse(bw_array, value), key))
    else:
        for key, value in suit_training_dict.items():
            heapq.heappush(diff_dict, (mse(bw_array, value), key))

    smallest_value, smallest_key = heapq.heappop(diff_dict)
    if smallest_value < 0.1:
        return smallest_key

    return 'Unknown'


def main():
    saved_rank = None
    saved_suit = None
    while True:
        ret, frame = vid.read()

        scanner = CardScanner(src=frame, DEBUG=False)
        res = scanner.run()

        if scanner.approx is not None:
            cv2.drawContours(frame, scanner.approx, -1, (0, 255, 0), 5)
        else:
            saved_rank = None
            saved_suit = None

        if res:
            cv2.imshow('res', scanner.info_gray)
            cv2.imshow('edge', scanner.edges)
            cv2.imshow('color', scanner.info_colored)
            test_rank = scanner.get_bounding_box_image(0)
            test_suit = scanner.get_bounding_box_image(1)
            try:
                saved_rank = classify(test_rank, 'rank', False)
                saved_suit = classify(test_suit, 'suit', False)
            except ValueError:
                cv2.imwrite("./debug.jpg", scanner.info_gray)
                print(scanner.info_gray.shape)
                scanner.show_debug(wait=True)

        cv2.putText(frame, saved_rank, (10, 50), font,
                    1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, saved_suit, (10, 100), font,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('frame', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break


if __name__ == '__main__':
    load_training_data()
    main()
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
    sys.exit(0)
