import cv2
import numpy as np
import glob

''' 
img_array = []
for i in range(553):
    filename = 'VideoMaterial/Positions/Final/Pos{}.jpg'.format(i)
    print(filename)
    img = cv2.imread(filename)
    img_cw_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    height, width, layers = img_cw_90.shape
    size = (width,height)
    img_array.append(img_cw_90)
 
 
out = cv2.VideoWriter('positions.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 20, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
'''


fnames = glob.glob('LandmarkIDImages/*.jpg')
for f in fnames:
    x = 'LandmarkIDImages/TMP/ID0'+f[-7:]
    img = cv2.imread(f)
    cv2.imwrite(x, img)
