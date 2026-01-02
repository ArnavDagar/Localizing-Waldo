import cv2
import numpy as np
import matplotlib.pyplot as plt

mmPerInch = 25.4
dpi = 200 # Printer dpi set to 200
pixelsPerMm = dpi / mmPerInch
pixelsPerCm = pixelsPerMm*10

pixelsC = round(127 * pixelsPerMm)
pixelsR = round(127 * pixelsPerMm)
img = np.zeros((pixelsR,pixelsC,3),np.uint8)
img += 255

p = 3

black = (0,0,0)
red = (255,0,0)
blue = (0,0,255)

def pmm(num):
    return round(pixelsPerMm*num)

cv2.rectangle(img, (0,0),(pixelsC,pixelsR), black, p)

cv2.rectangle(img, (pmm(3),pmm(3)), (pmm(23),pmm(23)), black, p)
cv2.rectangle(img, (pmm(127-23),pmm(3)), (pmm(127-3),pmm(23)), black, p)

cv2.rectangle(img, (pmm(19.5),pmm(31.5)), (pmm(19.5 + 88),pmm(31.5 + 64)), black, p)

cv2.circle(img, (pmm(63.5-10), pmm(31.5-20)), pmm(2.4), black, p)
cv2.circle(img, (pmm(63.5-10), pmm(31.5-20)), 0, red, p)
cv2.circle(img, (pmm(63.5-10+20), pmm(31.5-20)), pmm(2.4), black, p)
cv2.circle(img, (pmm(63.5-10+20), pmm(31.5-20)), 0, red, p)

cv2.circle(img, (pmm(63.5-10), pmm(31.5-20+104)), pmm(2.4), black, p)
cv2.circle(img, (pmm(63.5-10), pmm(31.5-20+104)), 0, red, p)
cv2.circle(img, (pmm(63.5-10+20), pmm(31.5-20+104)), pmm(2.4), black, p)
cv2.circle(img, (pmm(63.5-10+20), pmm(31.5-20+104)), 0, red, p)

cv2.circle(img, (pmm(44), pmm(127-3)), pmm(1.5), black, p)
cv2.circle(img, (pmm(44), pmm(127-3)), 0, red, p)
cv2.circle(img, (pmm(44+39), pmm(127-3)), pmm(1.5), black, p)
cv2.circle(img, (pmm(44+39), pmm(127-3)),0, red, p)

cv2.circle(img, (pmm(19.5+3), pmm(31.5+3)), pmm(1.5), black, p)
cv2.circle(img, (pmm(19.5+3), pmm(31.5+3)), 0, red, p)
cv2.circle(img, (pmm(19.5+3+82), pmm(31.5+3)), pmm(1.5), black, p)
cv2.circle(img, (pmm(19.5+3+82), pmm(31.5+3)), 0, red, p)
cv2.circle(img, (pmm(19.5+3), pmm(31.5+3+58)), pmm(1.5), black, p)
cv2.circle(img, (pmm(19.5+3), pmm(31.5+3+58)), 0, red, p)
cv2.circle(img, (pmm(19.5+3+82), pmm(31.5+3+58)), pmm(1.5), black, p)
cv2.circle(img, (pmm(19.5+3+82), pmm(31.5+3+58)), 0, red, p)

cv2.circle(img, (pmm(19.5+3-10), pmm(31.5+3)), pmm(1.5), black, p)
cv2.circle(img, (pmm(19.5+3-10), pmm(31.5+3)), 0, red, p)
cv2.circle(img, (pmm(19.5+3+82+10), pmm(31.5+3)), pmm(1.5), black, p)
cv2.circle(img, (pmm(19.5+3+82+10), pmm(31.5+3)), 0, red, p)
cv2.circle(img, (pmm(19.5+3-10), pmm(31.5+3+58)), pmm(1.5), black, p)
cv2.circle(img, (pmm(19.5+3-10), pmm(31.5+3+58)), 0, red, p)
cv2.circle(img, (pmm(19.5+3+82+10), pmm(31.5+3+58)), pmm(1.5), black, p)
cv2.circle(img, (pmm(19.5+3+82+10), pmm(31.5+3+58)), 0, red, p)

cv2.circle(img, (pmm(3+5), pmm(3+4)), pmm(1.5), black, p)
cv2.circle(img, (pmm(3+5), pmm(3+4)), 0, red, p)
cv2.circle(img, (pmm(3+5+10), pmm(3+4)), pmm(1.5), black, p)
cv2.circle(img, (pmm(3+5+10), pmm(3+4)), 0, red, p)
cv2.circle(img, (pmm(3+5), pmm(3+4+12)), pmm(1.5), black, p)
cv2.circle(img, (pmm(3+5), pmm(3+4+12)), 0, red, p)
cv2.circle(img, (pmm(3+5+10), pmm(3+4+12)), pmm(1.5), black, p)
cv2.circle(img, (pmm(3+5+10), pmm(3+4+12)), 0, red, p)

cv2.circle(img, (pmm(104+5), pmm(3+4)), pmm(1.5), black, p)
cv2.circle(img, (pmm(104+5), pmm(3+4)), 0, red, p)
cv2.circle(img, (pmm(104+5+10), pmm(3+4)), pmm(1.5), black, p)
cv2.circle(img, (pmm(104+5+10), pmm(3+4)), 0, red, p)
cv2.circle(img, (pmm(104+5), pmm(3+4+12)), pmm(1.5), black, p)
cv2.circle(img, (pmm(104+5), pmm(3+4+12)), 0, red, p)
cv2.circle(img, (pmm(104+5+10), pmm(3+4+12)), pmm(1.5), black, p)
cv2.circle(img, (pmm(104+5+10), pmm(3+4+12)), 0, red, p)

fileName = 'RobotLayout.jpg'
plt.imsave(fileName, img, dpi = 200)
