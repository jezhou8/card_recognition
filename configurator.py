import cv2
import numpy as np
import platform
import sys
import json


class Configurator:
    def __init__(self, config_output_file):
        cv2.setUseOptimized(True)
        self.config = {
            'x_offset': 0,
            'y_offset': 0,
            'x_width': 0,
            'y_width': 0,
        }
        self.config_set = False

        # define a video capture object
        if platform.system() == 'Windows':
            self.vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            self.vid = cv2.VideoCapture(0)

        print("Press P to take a Picture, Q to quit")

        self.window_name = "Cam View"
        self.preview_name = "Preview"
        cv2.namedWindow(winname=self.window_name)

    def exit(self):
        cv2.destroyAllWindows()
        self.vid.release()
        sys.exit()

    def draw_capture_area(self, event, x, y, flags, param):
        if self.config_set and (event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_LBUTTONUP):
            print("Press R to retry")
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            print('hello!')
            self.config['x_offset'] = x
            self.config['y_offset'] = y
            cv2.circle(param, (x, y), 3, (0, 255, 0), -1)
            cv2.imshow(self.window_name, param)

        if event == cv2.EVENT_LBUTTONUP:
            print('goodbye!')
            self.config['x_width'] = x - self.config['x_offset']
            self.config['y_width'] = y - self.config['y_offset']

            if self.config['x_width'] == 0 or self.config['y_width'] == 0:
                print("Invalid area")
                self.config_set = False
                return

            self.config_set = True
            param_clean = param.copy()
            cv2.rectangle(
                param, (self.config['x_offset'], self.config['y_offset']), (x, y), (0, 255, 0), 2)
            cv2.imshow(self.window_name, param)
            cv2.imshow(self.preview_name, param_clean[self.config['y_offset']:self.config['y_offset'] +
                                                      self.config['y_width'], self.config['x_offset']:self.config['x_offset'] + self.config['x_width']])

    def get_image(self):
        frame = None
        while True:
            _, frame = self.vid.read()

            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'):
                break
            if key == ord('q'):
                self.exit()

        return frame

    def save(self, frame=None):
        if frame is not None:
            cv2.imwrite('config_source.png', frame)

        self.config_file = open(config_output_file, 'w')
        self.config_file.write(json.dumps(self.config))
        self.config_file.close()
        print("Config Saved")

    def run(self):
        frame = self.get_image()
        frame_orig = frame.copy()
        cv2.setMouseCallback(
            self.window_name, self.draw_capture_area, frame)
        print("Press S to save, R to Retry, Q to quit")
        cv2.waitKey(0)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('s'):
            self.save(frame=frame_orig)
            self.exit()
        if key == ord('r'):
            self.config_set = False
            cv2.setMouseCallback(self.window_name, lambda *args: None)
            self.run()
        if key == ord('q'):
            self.exit()


if __name__ == "__main__":
    config_output_file = "config.json"
    configurator = Configurator(config_output_file)
    configurator.run()
