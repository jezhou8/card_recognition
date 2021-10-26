import cv2
import time
import os
import sys

from train import DEBUG, CardScanner


cv2.setUseOptimized(True)

# define a video capture object
vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)

prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

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
    saved_res = None
    saved_final_cnts = None
    while True:
        ret, frame = vid.read()

        scanner = CardScanner(src=frame, DEBUG=False)
        res = scanner.run()

        frame_to_show = frame
        if res:
            frame_to_show = scanner.og_image
            saved_res = scanner.info_gray
            saved_final_cnts = scanner.final_cnts
            cv2.imshow('res', scanner.info_gray)
            cv2.imshow('edge', scanner.edges)

        cv2.imshow('frame', frame_to_show)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            vid.release()
            cv2.destroyAllWindows()
            sys.exit(0)
        if key == ord('p'):
            bb = 0 if file_index < 12 else 1
            try:
                scanner.save_bounding_box(bb, file_path + file_name)
            except AttributeError:
                if saved_res is not None and saved_final_cnts is not None:
                    x, y, w, h = cv2.boundingRect(saved_final_cnts[bb])
                    cv2.imwrite(file_path+file_name, saved_res[y:y+h, x:x+w])
            break

    file_index += 1


# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
