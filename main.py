import cv2
import time
import os
import sys

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
print(file_path)
while file_index < len(file_names):
    file_name = f'{file_names[file_index]}.jpg'

    # Capture the video frame
    # by frame
    print(f"[{file_name}] Press 'p' to take picture', q to quit")
    saved_info_gray = None
    saved_final_cnts = None
    while True:
        ret, frame = vid.read()

        scanner = CardScanner(src=frame, DEBUG=False)
        res = scanner.run()

        if res:
            saved_info_gray = scanner.info_gray
            saved_final_cnts = scanner.final_cnts
            cv2.imshow('res', scanner.info_gray)
            cv2.imshow('edge', scanner.edges)
            cv2.imshow('color', scanner.info_colored)

        if scanner.approx is not None:
            cv2.drawContours(frame, scanner.approx, -1, (0, 255, 0), 5)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            vid.release()
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
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
