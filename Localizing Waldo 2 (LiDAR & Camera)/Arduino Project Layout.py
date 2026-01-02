import cv2
import numpy as np
import matplotlib.pyplot as plt

logo = plt.imread("ArdCampLogo.jpg")
logo90 = cv2.rotate(logo, cv2.ROTATE_90_CLOCKWISE)
utube = plt.imread("utube.jpg")
utube2 = plt.imread("utube2.jpg")


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

#Text

img[0:772, 0: 394] = logo90
#img[1150:1535, 1740:1930] = utube
img[900:1500, 1900:] = utube2
cv2.rectangle(img, (0,0),(pixelsC,pixelsR), black, p)

#Ultrasonic Sensor

cv2.line(img, (pmm(15),pmm(110)),(pmm(15),pmm(200)), black, p)
cv2.rectangle(img, (pmm(35),pmm(200)),(pmm(15), pmm(190)), black, p)

cv2.circle(img, (pmm(23),pmm(195)),pmm(1.5) , black, p)
cv2.circle(img, (pmm(23),pmm(195)),0 , red, p)

cv2.rectangle(img, (pmm(35),pmm(120)),(pmm(15), pmm(110)), black, p)

cv2.circle(img, (pmm(23),pmm(115)),pmm(1.5) , black, p)
cv2.circle(img, (pmm(23),pmm(115)),0 , red, p)

#Battery

cv2.rectangle(img, (pmm(91), pmm(98)), (pmm(68), pmm(45)), black, p)

cv2.circle(img, (pmm(68-2.5), pmm(71.5)), pmm(2.5), black, p)
cv2.circle(img, (pmm(68-2.5), pmm(71.5)), 0, red, p)

cv2.circle(img, (pmm(68+23+2.5), pmm(71.5)), pmm(2.5), black, p)
cv2.circle(img, (pmm(68+2.5+23), pmm(71.5)), 0, red, p)

cv2.circle(img, (pmm(79.5), pmm(98+2.5)), pmm(2.5), black, p)
cv2.circle(img, (pmm(79.5), pmm(98+2.5)), 0, red, p)

cv2.circle(img, (pmm(79.5), pmm(45-2.5)), pmm(2.5), black, p)
cv2.circle(img, (pmm(79.5), pmm(45-2.5)), 0, red, p)

cv2.putText(img, "9V B", (pmm(70.5), pmm(71.5)), cv2.FONT_HERSHEY_SIMPLEX, 2, red, p)

#Lamp
#cv2.rectangle(img, (pmm(79.5), pmm(25)), (pmm(49.5), pmm(15)), black, p)

cv2.circle(img, (pmm(79.5-6), pmm(20)), pmm(1), black, p)
cv2.circle(img, (pmm(79.5-6), pmm(20)), 0, red, p)

cv2.circle(img, (pmm(79.5-6-18), pmm(20)), pmm(1), black, p)
cv2.circle(img, (pmm(79.5-6-18), pmm(20)), 0, red, p)

cv2.putText(img, "L", (pmm(79.5-6-10.5), pmm(22)), cv2.FONT_HERSHEY_SIMPLEX, 2, red, p)

#Arduino

cv2.rectangle(img, (pmm(164), pmm(98)), (pmm(111), pmm(29)), black, p)

cv2.circle(img, (pmm(164-17.78), pmm(98-2.54)), pmm(1.6), black, p)
cv2.circle(img, (pmm(164-17.78), pmm(98-2.54)), 0, red, p)

cv2.circle(img, (pmm(111+2.54), pmm(29+15.24)), pmm(1.6), black, p)
cv2.circle(img, (pmm(111+2.54), pmm(29+15.24)), 0, red, p)

cv2.circle(img, (pmm(164-45.72), pmm(98-2.54)), pmm(1.6), black, p)
cv2.circle(img, (pmm(164-45.72), pmm(98-2.54)), 0, red, p)

cv2.circle(img, (pmm(111+2.54+48.26), pmm(29+15.24)), pmm(1.6), black, p)
cv2.circle(img, (pmm(111+2.54+48.26), pmm(29+15.24)), 0, red, p)

cv2.putText(img, "Arduino", (pmm(123), pmm(63.5)), cv2.FONT_HERSHEY_SIMPLEX, 2, red, p)

#Breadboard

cv2.rectangle(img, (pmm(239), pmm(98)), (pmm(184), pmm(15)), black, p)
cv2.putText(img, "Breadboard", (pmm(188), pmm(56.5)), cv2.FONT_HERSHEY_SIMPLEX, 2, red, p)

#TH Sensor

cv2.rectangle(img, (pmm(215), pmm(188)), (pmm(215-20), pmm(188-40)), black, p)
cv2.putText(img, "TH", (pmm(200), pmm(168)), cv2.FONT_HERSHEY_SIMPLEX, 2, red, p)

cv2.circle(img, (pmm(195), pmm(158)), pmm(1), black, p)
cv2.circle(img, (pmm(195), pmm(158)), 0, red, p)

cv2.circle(img, (pmm(215), pmm(158)), pmm(1), black, p)
cv2.circle(img, (pmm(215), pmm(158)), 0, red, p)

cv2.circle(img, (pmm(205), pmm(188)), pmm(1), black, p)
cv2.circle(img, (pmm(205), pmm(188)), 0, red, p)

#LED

cv2.rectangle(img, (pmm(175), pmm(188)), (pmm(175-20), pmm(188-20)), black, p)
cv2.putText(img, "LED", (pmm(158), pmm(181)), cv2.FONT_HERSHEY_SIMPLEX, 2, red, p)

cv2.circle(img, (pmm(155), pmm(178)), pmm(1), black, p)
cv2.circle(img, (pmm(155), pmm(178)), 0, red, p)

cv2.circle(img, (pmm(175), pmm(178)), pmm(1), black, p)
cv2.circle(img, (pmm(175), pmm(178)), 0, red, p)

#Buzzer

cv2.rectangle(img, (pmm(135), pmm(188)), (pmm(135-20), pmm(188-20)), black, p)
cv2.putText(img, "PB", (pmm(120), pmm(181)), cv2.FONT_HERSHEY_SIMPLEX, 2, red, p)

cv2.circle(img, (pmm(115), pmm(178)), pmm(1), black, p)
cv2.circle(img, (pmm(115), pmm(178)), 0, red, p)

cv2.circle(img, (pmm(135), pmm(178)), pmm(1), black, p)
cv2.circle(img, (pmm(135), pmm(178)), 0, red, p)

#Display

cv2.rectangle(img, (pmm(95), pmm(188)), (pmm(95-40), pmm(188-20)), black, p)
cv2.putText(img, "Display", (pmm(61.5), pmm(181)), cv2.FONT_HERSHEY_SIMPLEX, 2, red, p)

cv2.circle(img, (pmm(55+10), pmm(168)), pmm(1), black, p)
cv2.circle(img, (pmm(65), pmm(168)), 0, red, p)

cv2.circle(img, (pmm(55+10), pmm(188)), pmm(1), black, p)
cv2.circle(img, (pmm(65), pmm(188)), 0, red, p)

cv2.circle(img, (pmm(95), pmm(178)), pmm(1), black, p)
cv2.circle(img, (pmm(95), pmm(178)), 0, red, p)

#Pot

cv2.rectangle(img, (pmm(239), pmm(138)), (pmm(219), pmm(118)), black, p)
cv2.putText(img, "POT", (pmm(222), pmm(131)), cv2.FONT_HERSHEY_SIMPLEX, 2, red, p)

cv2.circle(img, (pmm(229), pmm(138)), pmm(1), black, p)
cv2.circle(img, (pmm(229), pmm(138)), 0, red, p)

cv2.circle(img, (pmm(229), pmm(118)), pmm(1), black, p)
cv2.circle(img, (pmm(229), pmm(118)), 0, red, p)

#Board Support

cv2.circle(img, (pmm(7.5), pmm(7.5)), pmm(1.25), black, p)
cv2.circle(img, (pmm(7.5), pmm(7.5)), 0, red, p)

cv2.circle(img, (pmm(254-7.5), pmm(7.5)), pmm(1.25), black, p)
cv2.circle(img, (pmm(254-7.5), pmm(7.5)), 0, red, p)

cv2.circle(img, (pmm(7.5), pmm(203-7.5)), pmm(1.25), black, p)
cv2.circle(img, (pmm(7.5), pmm(203-7.5)), 0, red, p)

cv2.circle(img, (pmm(254-7.5), pmm(203-7.5)), pmm(1.25), black, p)
cv2.circle(img, (pmm(254-7.5), pmm(203-7.5)), 0, red, p)

cv2.circle(img, (pmm(127), pmm(7.5)), pmm(1.25), black, p)
cv2.circle(img, (pmm(127), pmm(7.5)), 0, red, p)

cv2.circle(img, (pmm(127), pmm(203-7.5)), pmm(1.25), black, p)
cv2.circle(img, (pmm(127), pmm(203-7.5)), 0, red, p)

fileName = 'ArduinoLayout.jpg'
plt.imsave(fileName, img, dpi = 200)
