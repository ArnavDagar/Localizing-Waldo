import cv2
import numpy as np
import matplotlib.pyplot as plt

wpID = 0
mmPerInch = 25.4
dpi = 200 # Printer dpi set to 200
pixelsPerMm = dpi / mmPerInch
pixelsPerCm = pixelsPerMm*10

pixelsC = round(2210*pixelsPerMm)
pixelsR = round(2210*pixelsPerMm)
img = np.zeros((pixelsC,pixelsR,3),np.uint8)
img += 255

center = 1105
r = 915 # radius in mm of inner circle
a = 96
centerCoord = round(center*pixelsPerMm)
radius = round(r*pixelsPerMm)
outradius = round((r+a)*pixelsPerMm)
midradius = round((r+a/2)*pixelsPerMm)
topleft = centerCoord - radius
bottomright = centerCoord + radius
mm25 = round(pixelsPerMm*25)
mm12 = round(pixelsPerMm*12.5)
mm1 = round(pixelsPerMm)
mm3 = round(pixelsPerMm*3)
p = 5

black = (0,0,0)
blue = (0,0,255)
blue2 = (0, 61, 165)
white = (255, 255, 255)
white2 = (217,217,214)

wpL = np.array([center-900, center-450, center, center+450, center+900])*pixelsPerMm
#cL = np.array([center-600, center-300, center, center+300, center+600])*pixelsPerMm
cL = np.array([center-900, center-450, center, center+450, center+900])*pixelsPerMm
cLm = round((center-50)*pixelsPerMm)
cLl = round((center-100)*pixelsPerMm)

deltaOut = round((outradius+3)*np.sqrt(2)/2)
deltaIn = round((radius-mm12)*np.sqrt(2)/2)
delta = round((radius+mm12)*np.sqrt(2)/2)

c1ax = centerCoord-deltaOut
c1ay = centerCoord-deltaOut
c1bx = centerCoord-deltaIn
c1by = centerCoord-deltaIn
c1cx = centerCoord-delta
c1cy = centerCoord-delta

c2ax = centerCoord+deltaOut
c2ay = centerCoord-deltaOut
c2bx = centerCoord+deltaIn
c2by = centerCoord-deltaIn
c2cx = centerCoord+delta
c2cy = centerCoord-delta

c3ax = centerCoord-deltaOut
c3ay = centerCoord+deltaOut
c3bx = centerCoord-deltaIn
c3by = centerCoord+deltaIn
c3cx = centerCoord-delta
c3cy = centerCoord+delta

c4ax = centerCoord+deltaOut
c4ay = centerCoord+deltaOut
c4bx = centerCoord+deltaIn
c4by = centerCoord+deltaIn
c4cx = centerCoord+delta
c4cy = centerCoord+delta

#Add Way Point IDs
#Center Line
for c in cL:
    wpID += 1 
    wpIDText ='{}'.format(wpID)
    cv2.putText(img, wpIDText, (cLl-3*mm25,round(c)+mm25), cv2.FONT_HERSHEY_SIMPLEX, 24, white2, 2*mm1)
#Left Line
for wp in wpL:
    wpID += 1 
    wpIDText ='{}'.format(wpID)
    cv2.putText(img, wpIDText, (centerCoord-outradius-mm12-3*mm25,round(wp)+mm25), cv2.FONT_HERSHEY_SIMPLEX, 32, white2, 2*mm1)
#Right Line
for wp in wpL:
    wpID += 1 
    wpIDText ='{}'.format(wpID)
    cv2.putText(img, wpIDText, (centerCoord+outradius-mm25,round(wp)+mm25), cv2.FONT_HERSHEY_SIMPLEX, 24, white2, 2*mm1)
#Top Line
for wp in wpL:
    wpID += 1 
    wpIDText ='{}'.format(wpID)
    cv2.putText(img, wpIDText, (round(wp)-2*mm25,centerCoord-outradius-p-mm12), cv2.FONT_HERSHEY_SIMPLEX, 24, white2, 2*mm1)    
#Bottom Line
for wp in wpL:
    wpID += 1 
    wpIDText ='{}'.format(wpID)
    cv2.putText(img, wpIDText, (round(wp)-2*mm25,centerCoord+outradius+p+3*mm25), cv2.FONT_HERSHEY_SIMPLEX, 24, white2, 2*mm1)
#Circular Track
wpID += 1 
wpIDText ='{}'.format(wpID)
cv2.putText(img, wpIDText, (c1ax-5*mm25,c1ay-mm25), cv2.FONT_HERSHEY_SIMPLEX, 24, white2, 2*mm1)
#cv2.line(img, (c1ax,c1ay),(c1cx,c1cy), blue2, p)
wpID += 1 
wpIDText ='{}'.format(wpID)
cv2.putText(img, wpIDText, (c2ax+mm25,c2ay-mm25), cv2.FONT_HERSHEY_SIMPLEX, 24, white2, 2*mm1)
#cv2.line(img, (c2ax,c2ay),(c2cx,c2cy), blue2, p)
wpID += 1 
wpIDText ='{}'.format(wpID)
cv2.putText(img, wpIDText, (c3ax-5*mm25,c3ay+3*mm25), cv2.FONT_HERSHEY_SIMPLEX, 24, white2, 2*mm1)
#cv2.line(img, (c3ax,c3ay),(c3cx,c3cy), blue2, p)
wpID += 1 
wpIDText ='{}'.format(wpID)
cv2.putText(img, wpIDText, (c4ax+mm25,c4ay+3*mm25), cv2.FONT_HERSHEY_SIMPLEX, 24, white2, 2*mm1)
#cv2.line(img, (c4ax,c4ay),(c4cx,c4cy), blue2, p)

cv2.circle(img, (centerCoord,centerCoord),radius , black, mm25)
cv2.circle(img, (centerCoord,centerCoord),radius, white, mm1)
cv2.circle(img, (centerCoord,centerCoord),midradius, black, p)
cv2.circle(img, (centerCoord,centerCoord),outradius, black, p)

#cv2.circle(img, (centerCoord,centerCoord), round(930*pixelsPerMm), black, 10)
cv2.rectangle(img, (topleft,topleft),(bottomright, bottomright),black,mm25)
cv2.rectangle(img, (topleft,topleft),(bottomright, bottomright),white,mm1)
cv2.rectangle(img, (centerCoord-midradius,centerCoord-midradius),(centerCoord+midradius, centerCoord+midradius),black,p)
cv2.rectangle(img, (centerCoord-outradius,centerCoord-outradius),(centerCoord+outradius, centerCoord+outradius),black,p)

cv2.line(img, (centerCoord, round(cL[0])), (centerCoord, round(cL[4])), black, mm25)
cv2.line(img, (centerCoord, round(cL[0])), (centerCoord, round(cL[4])), white, p)
cv2.line(img, (cLm, round(cL[0])), (cLm, round(cL[4])), black, p)
cv2.line(img, (cLl, round(cL[0])), (cLl, round(cL[4])), black, p)

#Draw Way Points: Center Line
for c in cL:
    cv2.line(img, (cLl-p,round(c)), (centerCoord+mm12,round(c)), white, p)
    cv2.line(img, (cLl-p,round(c)), (centerCoord-mm12,round(c)), blue, mm3)
    cv2.line(img, (cLl-p,round(c)), (centerCoord-mm12,round(c)), blue2, p)
    
#Draw Way Points: Outer Lines
for wp in wpL:
    #Left Line
    cv2.line(img, (centerCoord-outradius-p, round(wp)), (topleft+mm25, round(wp)), white, p)
    cv2.line(img, (centerCoord-outradius-p, round(wp)), (topleft-mm12, round(wp)), blue, mm3)
    cv2.line(img, (centerCoord-outradius-p, round(wp)), (topleft-mm12, round(wp)), blue2, p)
    #Right Line
    cv2.line(img, (bottomright-mm25, round(wp)), (centerCoord+outradius+p, round(wp)), white, p)
    cv2.line(img, (bottomright+mm12, round(wp)), (centerCoord+outradius+p, round(wp)), blue, mm3)
    cv2.line(img, (bottomright+mm12, round(wp)), (centerCoord+outradius+p, round(wp)), blue2, p)
    #Top Line
    cv2.line(img, (round(wp),centerCoord-outradius-p), (round(wp),topleft+mm25), white, p)
    cv2.line(img, (round(wp),centerCoord-outradius-p), (round(wp),topleft-mm12), blue, mm3)
    cv2.line(img, (round(wp),centerCoord-outradius-p), (round(wp),topleft-mm12), blue2, p)
    #Bottom Line
    cv2.line(img, (round(wp),bottomright-mm25), (round(wp),centerCoord+outradius+p), white, p)
    cv2.line(img, (round(wp),bottomright+mm12), (round(wp),centerCoord+outradius+p), blue, mm3)
    cv2.line(img, (round(wp),bottomright+mm12), (round(wp),centerCoord+outradius+p), blue2, p)

cv2.line(img, (c1ax,c1ay),(c1bx,c1by), white, p)
cv2.line(img, (c2ax,c2ay),(c2bx,c2by), white, p)
cv2.line(img, (c3ax,c3ay),(c3bx,c3by), white, p)
cv2.line(img, (c4ax,c4ay),(c4bx,c4by), white, p)

cv2.line(img, (c1ax,c1ay),(c1cx,c1cy), blue, mm3)
cv2.line(img, (c2ax,c2ay),(c2cx,c2cy), blue, mm3)
cv2.line(img, (c3ax,c3ay),(c3cx,c3cy), blue, mm3)
cv2.line(img, (c4ax,c4ay),(c4cx,c4cy), blue, mm3)

cv2.line(img, (c1ax,c1ay),(c1cx,c1cy), blue2, p)
cv2.line(img, (c2ax,c2ay),(c2cx,c2cy), blue2, p)
cv2.line(img, (c3ax,c3ay),(c3cx,c3cy), blue2, p)
cv2.line(img, (c4ax,c4ay),(c4cx,c4cy), blue2, p)

cSquare = np.array([center - 605, center + 605])*pixelsPerMm
cSquareC1 = round(cSquare[0])
cSquareC2 = round(cSquare[1])
#cv2.rectangle(img, (cSquareC1, cSquareC1), (cSquareC2, cSquareC2), black, p)

def generate_pages(Bx,By,bID):
    count = 0
    for i in range(len(Bx)-1):
        for j in range(len(By)-1):
            count+=1
            sx = round(Bx[i]);
            ex = round(Bx[i+1])
            sy = round(By[j])
            ey = round(By[j+1])
            # print(i,j,sx,ex,sy,ey)
            page = img[sy:ey,sx:ex:]
            # print(page.shape)
            cv2.rectangle(page, (0,0),(page.shape[1],page.shape[0]), black, 2)
            fileName = 'Block {}_{}.jpg'.format(bID, count) 
            plt.imsave(fileName, page, dpi = 200)
plt.imshow(img)
plt.show(block = False)

B1x = np.array([0, 250, 500])*pixelsPerMm
B1y = np.array([0, 100, 300, 500])*pixelsPerMm

B2x = np.array([500, 700, 900, 1105, 1305, 1505, 1710])*pixelsPerMm
B2y = np.array([0, 250, 500])*pixelsPerMm

B3x = np.array([1710, 1960, 2210])*pixelsPerMm
B3y = np.array([0, 100, 300, 500])*pixelsPerMm

B4x = np.array([0, 250, 500])*pixelsPerMm
B4y = np.array([500, 705, 905, 1105, 1305, 1505, 1710])*pixelsPerMm

B5x = np.array([900, 1155])*pixelsPerMm
B5y = np.array([500, 705, 905, 1105, 1305, 1505, 1710])*pixelsPerMm

B6x = np.array([1710, 1960, 2210])*pixelsPerMm
B6y = np.array([500, 705, 905, 1105, 1305, 1505, 1710])*pixelsPerMm

B7x = np.array([0, 250, 500])*pixelsPerMm
B7y = np.array([1710, 1910, 2110, 2210])*pixelsPerMm

B8x = np.array([500, 700, 900, 1105, 1305, 1505, 1710])*pixelsPerMm
B8y = np.array([1710, 1960, 2210])*pixelsPerMm

B9x = np.array([1710, 1960, 2210])*pixelsPerMm
B9y = np.array([1710, 1910, 2110, 2210])*pixelsPerMm

generate_pages(B1x,B1y,1)
generate_pages(B2x,B2y,2)
generate_pages(B3x,B3y,3)
generate_pages(B4x,B4y,4)
generate_pages(B5x,B5y,5)
generate_pages(B6x,B6y,6)
generate_pages(B7x,B7y,7)
generate_pages(B8x,B8y,8)
generate_pages(B9x,B9y,9)
