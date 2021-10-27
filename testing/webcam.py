import cv2
import numpy as np
import time
import os

# define a video capture object
vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)

prev_frame_time = 0
 
# used to record the time at which we processed current frame
new_frame_time = 0
font = cv2.FONT_HERSHEY_SIMPLEX

ret, frame = vid.read()
print('Resolution: ' + str(frame.shape[0]) + ' x ' + str(frame.shape[1]))
while(True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    new_frame_time = time.time()
 
    # Calculating the fps
 
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
 
    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)
 
    # putting the FPS count on the frame
    cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    # Display the resulting frame
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
