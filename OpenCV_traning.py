import cv2
import numpy as np

#image
# img = cv2.imread("image.png",1) #read img
# cv2.namedWindow("test_image",cv2.WINDOW_NORMAL)
# cv2.imshow("test_image",img) #show img
# cv2.waitKey(0) == 27 # close windows after pressing 'Esc'
# cv2.destroyAllWindows() #close all windows

#draw rectangle
img = cv2.rectangle(img,(0,0,255),720)

#video
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow("Video", frame)

    if (cv2.waitKey(30) == 27):
        break

cv2.destroyAllWindows() #close all windows