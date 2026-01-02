import cv2
from picamera import PiCamera
import time

camera = PiCamera()
camera.resolution = (1640, 1232)
camera.start_preview(fullscreen = False, window = (10,20, 410, 308))
#time.sleep(15)

img_counter = 0
while True:
    print("here")
    k = int(input().strip())
    print(k)
    if k== 1: 
        print("Closing...")
        break
    elif k == 2:
        img_name = "Data/CalImgs2/Test{}.jpg".format(img_counter)
        camera.capture(img_name)
        print("{} capture!".format(img_name))
        img_counter += 1
        
camera.stop_preview()