import cv2
import numpy as np
import matplotlib.pyplot as plt

logo = plt.imread("ArdCampLogo.jpg")
logo90 = cv2.rotate(logo, cv2.ROTATE_90_CLOCKWISE)
utube = plt.imread("utube.jpg")

mmPerInch = 25.4
dpi = 200 # Printer dpi set to 200
pixelsPerMm = dpi / mmPerInch
pixelsPerCm = pixelsPerMm*10

pixelsC = round(10*dpi)
pixelsR = round(8*dpi)
img = np.zeros((pixelsR,pixelsC,3),np.uint8)
img += 255

def pmm(num):
    return round(pixelsPerMm*num)

gUS = (50,25)
gLED = (20,20)
gTH = (40,20)
gPB = (20,20)
gOD = (20,40)
gPT = (20,20)
gBH = (58, 32)
gB9 = (53,23)
gAB = (69,53)
gBB = (83,55)

p = 3

black = (0,0,0)
red = (255,0,0)
blue = (0,0,255)

def createUS(offsetX, offsetY):
    cv2.rectangle(img, (pmm(offsetX+90),pmm(offsetY+50)),(pmm(offsetX+0),pmm(offsetY+0)), black, p)
   # cv2.rectangle(img, (pmm(offsetX+10),pmm(offsetY+20)),(pmm(offsetX+0),pmm(offsetY+0)), black, p)
   # cv2.rectangle(img, (pmm(offsetX+90),pmm(offsetY+20)),(pmm(offsetX+80),pmm(offsetY+0)), black, p)
    cv2.rectangle(img, (pmm(offsetX+70),pmm(offsetY+40)),(pmm(offsetX+20),pmm(offsetY+15)), black, p)

    cv2.circle(img, (pmm(offsetX+5),pmm(offsetY+8)),pmm(1.5) , black, p)
    cv2.circle(img, (pmm(offsetX+5),pmm(offsetY+8)),0 , red, p)

    cv2.circle(img, (pmm(offsetX+85),pmm(offsetY+8)),pmm(1.5) , black, p)
    cv2.circle(img, (pmm(offsetX+85),pmm(offsetY+8)),0 , red, p)

    cv2.circle(img, (pmm(offsetX+21.25+0.25),pmm(offsetY+27.5)),pmm(1.25) , black, p)
    cv2.circle(img, (pmm(offsetX+21.25+0.25),pmm(offsetY+27.5)),0 , red, p)

    cv2.circle(img, (pmm(offsetX+52.5),pmm(offsetY+16.25+0.25)),pmm(1.5) , black, p)
    cv2.circle(img, (pmm(offsetX+52.5),pmm(offsetY+16.25+0.25)),0 , red, p)

    cv2.circle(img, (pmm(offsetX+52.5),pmm(offsetY+40-1.25-0.25)),pmm(1.5) , black, p)
    cv2.circle(img, (pmm(offsetX+52.5),pmm(offsetY+40-1.25-0.25)),0 , red, p)

createUS(0,0)
createUS(0,60)
createUS(0,120)

createUS(100,0)
createUS(100,60)
createUS(100,120)


fileName = 'USLayout.jpg'
plt.imsave(fileName, img, dpi = 200)
