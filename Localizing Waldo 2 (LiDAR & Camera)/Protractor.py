import cv2
import numpy as np
import matplotlib.pyplot as plt

mmPerInch = 25.4
dpi = 200 # Printer dpi set to 200
pixelsPerMm = dpi / mmPerInch
pixelsPerCm = pixelsPerMm*10

pixelsC = round(200*pixelsPerMm)
pixelsR = round(200*pixelsPerMm)

img = np.zeros((pixelsC,pixelsR,3),np.uint8)
img += 255

center = 100
r1 = 80
r2 = 85
r3 = 90
centerCoord = round(center*pixelsPerMm)
outR = round(r2*pixelsPerMm)
inR = round(r1*pixelsPerMm)
txtR = round(r3*pixelsPerMm)

black = (0,0,0)
p = 3

'''
for i in range(36):
    angle = (10*i+5)*np.pi/180
    x = centerCoord-round(inR*np.cos(angle))
    y = centerCoord-round(inR*np.sin(angle))
    cv2.line(img, (centerCoord, centerCoord),(x,y), black, p)
'''
for i in range(36):
    angle = 10*i*np.pi/180
    x = centerCoord-round(outR*np.cos(angle))
    y = centerCoord-round(outR*np.sin(angle))
    aTxt = '{}'.format(36-i)
    x1 = centerCoord-round(txtR*np.cos(angle))
    y1 = centerCoord-round(txtR*np.sin(angle))
    cv2.line(img, (centerCoord, centerCoord),(x,y), black, p)
    cv2.putText(img, aTxt, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, black, 1)
cv2.line(img, (0, centerCoord),(pixelsR,centerCoord), black, p)
cv2.line(img, (centerCoord, 0),(centerCoord,pixelsC), black, p)
cv2.rectangle(img, (0,0),(pixelsC, pixelsR),black,p)
#cv2.circle(img, (centerCoord, centerCoord),round(7*pixelsPerMm),(255,255,255),20)

plt.imsave('protractor.jpg', img, dpi = 200)
plt.imshow(img)
plt.show(block = False)  
    
