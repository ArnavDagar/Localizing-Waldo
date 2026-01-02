'''ParticleFilter Module
Implements
1. Particle Filter
2. Functions to visualize and evaluate the performance
of ultrasonic and color sensors
3. Functions to estimate the measurements of these sensors
4. First cut - implements using python lists and for loops'''

from tkinter import *
import random
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.image as mpimg
import cv2 as cv

random.seed(2222022)
plt.rcParams["figure.figsize"] = (5,5)

def Gaussian(self, mu, sigma, x):
    '''Gaussian(self, mu, sigma, x)-> prob curve
    calculates the probability of x for 1-dim Gaussian
    with mean mu and var. sigma
    Source Udacity'''
    return exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / sqrt(2.0 * pi * (sigma ** 2))

def create_3_digit(num):
    '''create_3_digit(num)-> 3 digit number for a file name
    Assumes given numbers have 3 or less digits '''
    num = str(num)
    if len(num) == 3:
        return str(num)
    if len(num) == 2:
        return '0' + num
    if len(num) == 1:
        return '00' + num


def color_transition(marker):
    ''' color_transition(marker)-> arr
    generate the color transition array used for color strips'''
    finalList = []
    for i in range(7): # 7 colors
        if i+1 != marker:
            finalList.insert(0,marker)
            finalList.insert(0,i+1)
    finalList.insert(0,marker)
    
    arr = np.array(finalList)
    
    return arr

def true_color_strip(color_trans, dist, stripSize):
    '''true_color_strip(color_trans, dist, stripSize) -> color
    Finds the true color on a color strip image given its size and
    distance from a starting point'''
    startDist = 15
    #startDist = 14
    if dist < startDist:
        return 6
    q = int((dist-startDist)//stripSize)
    if q >= 13:
        return 6
    return color_trans[q]

def eval_CS_perf_strips():
    '''  eval_CS_perf_strips()-> None
    Color Sensor performance evaluation. Generates confusion matrices
    and images to visualize performance
    Goes through all the files and compares measurement with
    estimated truth '''
    
    fileList = ['CSC07', 'CSC08', 'CSC09', 'CSC10','CSC11', 'CSC12']
    stripSize = 25
    markers = [7, 5, 4, 3, 2, 1]
    marker_colors = ['BROWN','RED','YELLOW','GREEN','BLUE','BLACK']
    cm = np.zeros((8,8),dtype = int)
    
    colorCode = {} # Dictionary of rgb values for different colors
    colorCode[0] = (0, 0, 0)
    colorCode[1] = (26.0/255.0, 26.0/255.0, 26.0/255.0)
    colorCode[2] = (0, 0, 255.0/255.0)
    colorCode[3] = (0, 150.0/255.0, 57.0/255.0)
    colorCode[4] = (255.0/255.0, 255.0/255.0, 0)
    colorCode[5] = (255.0/255.0, 0, 0)
    colorCode[6] = (255.0/255.0, 255.0/255.0, 255.0/255.0)
    colorCode[7] = (96.0/255.0, 61.0/255.0, 32.0/255.0)

    COLOR_MAP = {
            'None': 0,
            'Color.BLACK': 1,
            'Color.BLUE': 2,
            'Color.GREEN': 3,
            'Color.YELLOW': 4,
            'Color.RED': 5,
            'Color.WHITE': 6,
            'Color.BROWN': 7
            }

    for k in range(6):
        #fig, ax = plt.subplots()
        bbox = {'fc': '0.8', 'pad': 0}
        fig, axs = plt.subplots(2, 1, sharex=True)
        titleStr = 'Color Sensor Performance - Printed Vinyl, Day Time, ' + marker_colors[k] + ' Seperators' 
        fig.suptitle(titleStr,fontsize=12)
        #addtext(axs[1], {'ha': 'center', 'va': 'center', 'bbox': bbox})
        axs[1].set_ylabel('True Colors')
        #addtext(axs[0], {'ha': 'left', 'va': 'bottom', 'bbox': bbox})
        axs[0].set_ylabel('Sensed Colors')

        plt.show(block = False)
        # axs[0].axes.yaxis.set_visible(False)
        axs[0].set_yticklabels([])
        axs[0].set_yticks([])
        axs[1].set_yticklabels([])
        axs[1].set_yticks([])
        axs[1].set_xlabel('mm')
        #ax.annotate('Truth', xy = (0,135))
        #ax.annotate('Measured',xy = (0, 55))             
        countTrue = 0
        countFalse = 0
        countNone = 0
        fileStr = "C:/Users/Arnav/OneDrive/Documents/2022 Science Fair Project/Data/CSC/" +  fileList[k] + '.csv'
        marker = markers[k]
        color_trans = color_transition(marker)
        #print(color_trans)
        df1 = pd.read_csv(fileStr)
        #for i in range(nLines):
        for i in range(0,len(df1),5):
            dist = df1[' state[0]'][i] 
            meas = df1[' color'][i].strip()
            stripSize = (max(df1[' state[0]'])-min(df1[' state[0]'])-28)/13
            stripSize = 24.5
            #print(stripSize)
            truth = true_color_strip(color_trans, dist,stripSize)
            #print(truth)
            xCoords = [dist, dist]
            yCoords = [30, 80]   
            axs[1].plot(xCoords, yCoords, color = colorCode[truth], linewidth = 2)
            #ax.set_aspect('equal')
            #yCoords = [30, 80]
            cm[COLOR_MAP[meas], truth] += 1
            if meas == 'None':
                #print('got none')
                countNone += 1
            else:
                axs[0].plot(xCoords, yCoords, color = colorCode[COLOR_MAP[meas]], linewidth = 2)
                if COLOR_MAP[meas] == truth:
                    countTrue += 1
                else:
                    countFalse += 1
        plt.savefig(fileList[k] + '.png')
        #print('color'+'\t'+'Count True'+'\t' + 'Count False'+'\t' + 'Count None')               
        print(marker_colors[k], countTrue, countFalse, countNone)
        
    print(cm)
        
def set_figure(pL, ww, cwOffset, pCoords, pThetas):
    ''' Generate plots of particles and Waldo World'''
    fig, ax = plt.subplots()
    ax.plot(pL[0].get_TT_coords()[0]+cwOffset[0],pL[0].get_TT_coords()[1]+cwOffset[1], 'ko',markersize = 6)
    #ax.scatter(pL[0].get_CS_coords()[0]+cwOffset[0],pL[0].get_CS_coords()[1]+cwOffset[1])
    ax.plot(ww.get_usl_xCoords(),ww.get_usl_yCoords())
    ax.plot(ww.get_cb_xCoords(), ww.get_cb_yCoords())
    ax.plot(ww.get_cw_xCoords(), ww.get_cw_yCoords())
    ax.quiver(get_xCoords(pCoords,cwOffset), get_yCoords(pCoords,cwOffset), get_xdir_list(pThetas), get_ydir_list(pThetas))
    
    ax.set_aspect('equal')
    ax.set_xlim(left = 830, right = 980)
    ax.set_ylim(bottom = 800, top = 960)
    #ax.set_xlim(right = 1900)
    #ax.set_ylim(top = 1900)
    plt.show(block = False)

    ax.set_title('Color Sensor Measurements, Test Site 36', fontsize = 10)
    ax.set_xlabel('mm')
    ax.set_ylabel('mm')
    return ax
    
def eval_CS_perf(testID):
    ''' Visualizes and generates metrics for color sensor performance
    in color world during localization
    Works by creating a particle at test site and computing true colors
    at the color sensor coordinates'''
    ww = WaldoWorld()
    cw = ColorWorld(ww)
    fileStr = "C:/Users/Arnav/OneDrive/Documents/2022 Science Fair Project/Data/Test" + create_3_digit(testID) + ".csv"
    nLines = 104

    colorCode = {}
    colorCode[0] = (0, 0, 0)
    colorCode[1] = (26.0/255.0, 26.0/255.0, 26.0/255.0)
    colorCode[2] = (0, 0, 255.0/255.0)
    colorCode[3] = (0, 150.0/255.0, 57.0/255.0)
    colorCode[4] = (255.0/255.0, 255.0/255.0, 0)
    colorCode[5] = (255.0/255.0, 0, 0)
    colorCode[6] = (255.0/255.0, 255.0/255.0, 255.0/255.0)
    colorCode[7] = (96.0/255.0, 61.0/255.0, 32.0/255.0)

    COLOR_MAP = {
            'None': 0,
            'Color.BLACK': 1,
            'Color.BLUE': 2,
            'Color.GREEN': 3,
            'Color.YELLOW': 4,
            'Color.RED': 5,
            'Color.WHITE': 6,
            'Color.BROWN': 7
            }
    
    # Only 1 particle
    pf = ParticleFilter(1,ww) 
    pL = pf.get_particle_list()

    testCoords = cw.get_test_coords(testID)
    testHeading = cw.get_test_heading(testID)

    # Set the coordinates of the particle to test coords
    pL[0].set_coords(testCoords)
    pL[0].set_theta(testHeading)
    pL[0].set_phi(90)
    print(testCoords,testHeading)

    pCoords = pf.get_particle_coords_list()
    cwOffset = ww.get_cw_offset()
    pThetas = pf.get_particle_theta_list()

    countHit = 0
    countMiss = 0

    ax=set_figure(pL, ww, cwOffset, pCoords, pThetas)
    df1 = pd.read_csv(fileStr)
    #for i in range(nLines):
    for i in range(0,nLines,4):
        #ax=set_figure(pL, ww, cwOffset, pCoords, pThetas)
        phi = df1[' tt_angle'][i] * 180/205 + 90
        meas = df1[' color'][i].strip()
        pL[0].set_phi(phi)
        color = cw.find_color_true(pL[0].get_CS_coords())
        if color[0] == 'Color.WHITE':
        #print(colorCode[COLOR_MAP[color[0]]])
            ax.plot(pL[0].get_CS_coords()[0]+cwOffset[0],pL[0].get_CS_coords()[1]+cwOffset[1], marker='o',
                    markerfacecolor= colorCode[COLOR_MAP[color[0]]], markersize=12,markeredgewidth=0.5, markeredgecolor=(0, 0, 0))
        else:    
            ax.plot(pL[0].get_CS_coords()[0]+cwOffset[0],pL[0].get_CS_coords()[1]+cwOffset[1], marker='o',
                    markerfacecolor= colorCode[COLOR_MAP[color[0]]], markersize=12,markeredgewidth=0.5, markeredgecolor=colorCode[COLOR_MAP[color[0]]])

        if meas != None:
            if meas == 'Color.WHITE':
                ax.plot(pL[0].get_CS_coords()[0]+cwOffset[0],pL[0].get_CS_coords()[1]+cwOffset[1], marker='*',
                     markerfacecolor= colorCode[COLOR_MAP[meas]], markersize= 10, markeredgewidth=0.5, markeredgecolor=(0, 0, 0))
            else:
                ax.plot(pL[0].get_CS_coords()[0]+cwOffset[0],pL[0].get_CS_coords()[1]+cwOffset[1], marker='*',
                     markerfacecolor= colorCode[COLOR_MAP[meas]], markersize= 10, markeredgewidth=0.5, markeredgecolor=(0,0,0))      
        if meas in color:
            countHit += 1
            #ax.set_title('Color Sensor Measurement in Color World (Hit)', fontsize = 10)
        else:
            countMiss += 1
            #ax.set_title('Color Sensor Measurement in Color World (Miss)', fontsize = 10)
        #figStr = 'Color'+str(i)+'.jpg'
        figStr = 'CS36.jpg'
        plt.savefig(figStr)  
        print(i,phi,meas,color)

    print(countHit, countMiss)
    

def eval_US_perf(testID):
    ''' Visualizes and generates metrics for ultrasonic sensor performance
    in waldo world during localization
    Works by creating a particle at test site and computing
    expected distances, angles and what coordinates the ultrasonic sensor
    is aiming at a wall'''
    ww = WaldoWorld()
    cw = ColorWorld(ww)
    fileStr = "C:/Users/Arnav/OneDrive/Documents/2022 Science Fair Project/Data/Test" + create_3_digit(testID) + ".csv"
    print(fileStr)
    nLines = 104

    # i Partile
    pf = ParticleFilter(1,ww)
    pL = pf.get_particle_list()
    usl = ww.get_us_landmarks()

    testCoords = cw.get_test_coords(testID)
    testHeading = cw.get_test_heading(testID)

    # set particle coordinates to test coordinates
    pL[0].set_coords(testCoords)
    pL[0].set_theta(testHeading)
    pL[0].set_phi(90)
    print(testCoords,testHeading)
    print(pL[0])

    ttCoords = pL[0].get_TT_coords()
    usOffset = pL[0].get_offsetUS()
    print(ttCoords)
    pCoords = pf.get_particle_coords_list()
    cwOffset = ww.get_cw_offset()
    pThetas = pf.get_particle_theta_list()

    '''
    This code was used to generate frames of reference picture
    
    fig, ax = plt.subplots()
    #ax.scatter(pL[0].get_TT_coords()[0]+cwOffset[0],pL[0].get_TT_coords()[1]+cwOffset[1])
    #ax.scatter(pL[0].get_CS_coords()[0]+cwOffset[0],pL[0].get_CS_coords()[1]+cwOffset[1])
    ax.plot(ww.get_usl_xCoords(),ww.get_usl_yCoords())
    ax.plot(ww.get_cb_xCoords(), ww.get_cb_yCoords())
    ax.plot(ww.get_cw_xCoords(), ww.get_cw_yCoords())
    #ax.quiver(get_xCoords(pCoords,cwOffset), get_yCoords(pCoords,cwOffset), get_xdir_list(pThetas), get_ydir_list(pThetas))
    ax.plot([0,250],[0,0],'k')
    ax.quiver([250], [0], [500], [0])
    ax.annotate('x',xy = (325,20))
    ax.annotate('y',xy = (-75,325))

    ax.plot([0,0],[0,250],'k')
    ax.quiver([0], [250], [0], [1])
    ax.annotate('Map Coordinates', xy = (-75,-75))

    ax.plot([ww.get_cw_xCoords()[0],ww.get_cw_xCoords()[0]+250] , [ww.get_cw_yCoords()[0],ww.get_cw_yCoords()[0]],'k')
    ax.quiver(ww.get_cw_xCoords()[0]+250, ww.get_cw_yCoords()[0], 1, 0)
    ax.plot([ww.get_cw_xCoords()[0],ww.get_cw_xCoords()[0]] , [ww.get_cw_yCoords()[0],ww.get_cw_yCoords()[0]+250],'k')
    ax.quiver(ww.get_cw_xCoords()[0], ww.get_cw_yCoords()[0]+250, 0, 1)
    ax.annotate('Color World Coordinates', xy = (100,100))
    ax.annotate('x',xy = (600,225))
    ax.annotate('y',xy = (300,500))
    ax.annotate('x',xy = (600,1325))
    ax.annotate('columns',xy = (600,1250))
    ax.annotate('y',xy = (300,1075))
    ax.annotate('rows',xy = (150,950),rotation= -90)

    ax.plot([750,950],[750,850],'k')
    #ax.plot([750,1250],[750,1000],'k-.')
    ax.plot([750,1200],[750,750],'k-.')
    ax.annotate('Waldo Coordinates',xy = (650, 650))
    ax.annotate('Color World',xy = (500, 500))
    ax.quiver([950], [850], [2], [1])
    ax.annotate('x',xy = (950, 900), rotation = 26)
    ax.plot([750,650],[750,950],'k')
    ax.quiver([650], [950], [-1], [2])
    ax.annotate('y',xy = (500, 1000), rotation = 116)
    ax.annotate(r'$\theta$',xy = (940,775))

    ax.plot([ww.get_cw_xCoords()[1],ww.get_cw_xCoords()[1]+250] , [ww.get_cw_yCoords()[1],ww.get_cw_yCoords()[1]],'k')
    ax.quiver(ww.get_cw_xCoords()[1]+250, ww.get_cw_yCoords()[1], 1, 0)
    ax.plot([ww.get_cw_xCoords()[1],ww.get_cw_xCoords()[1]] , [ww.get_cw_yCoords()[1],ww.get_cw_yCoords()[1]-250],'k')
    ax.quiver(ww.get_cw_xCoords()[1], ww.get_cw_yCoords()[1]-250, 0, -1)
    ax.annotate('Image Coordinates', xy = (300,1420))
    
    print('xcords ', get_xCoords(pCoords,cwOffset))

    ax.set_aspect('equal')
    ax.set_xlim(right = 2000)
    ax.set_ylim(top = 2000)
    plt.show(block = False)

    ax.set_title('Waldo World')
    ax.set_xlabel('mm')
    ax.set_ylabel('mm')
    '''
    ax=set_figure(pL, ww, cwOffset, pCoords, pThetas)
    df1 = pd.read_csv(fileStr)
    for i in range(0,nLines,4):
    #for i in [0]:
        #ax=set_figure(pL, ww, cwOffset, pCoords, pThetas)
        phi = df1[' tt_angle'][i] * 180/205 + 90
        meas = df1[' distance'][i]
        print(meas)
        pL[0].set_phi(phi)
        #ax.scatter(pL[0].get_CS_coords()[0]+cwOffset[0],pL[0].get_CS_coords()[1]+cwOffset[1])
        alpha = pL[0].get_sensor_heading()
        for j in range(8):
            distance = usl[j].find_distance(pL[0],cwOffset)
            if distance != None:
                ax.plot([ttCoords[0] + cwOffset[0], distance[3][0]], [ttCoords[1] + cwOffset[1], distance[3][1]], color = 'blue', linewidth = 3)
        if meas != 2550:
            meas += usOffset
            #if 45 <= alpha <= 135:
            #    meas += 80
            #
            xl = ttCoords[0] + meas * np.cos(alpha*np.pi/180)
            yl = ttCoords[1] + meas * np.sin(alpha*np.pi/180)
            ax.plot(get_xCoords([ttCoords, (xl,yl)],cwOffset), get_yCoords([ttCoords, (xl,yl)],cwOffset), color = 'orange', linewidth = 1)
        else:
            print(meas, i)
        #ax.set_title('Ultrasonic Sensor Measurement in Waldo World', fontsize = 10)
        #figStr = 'US'+str(i)+'.jpg'
        #plt.savefig(figStr)  

    print(alpha, 'done')
    ax.set_title('Ultrasonic Sensor Measurements, Test Site 36', fontsize = 10)
    figStr = 'US'+str(testID)+'.jpg'
    print(figStr)
    plt.savefig(figStr)  



def get_xCoords(coordList, offset = (0,0)):
    '''get_xCoords(coordList, offset = (0,0)) -> x_values
    generate xcoords from a list of (xcoords, ycoords)'''
    x_values = [coord[0]+offset[0] for coord in coordList]
    return x_values

def get_yCoords(coordList, offset = (0,0)):
    '''get_yCoords(coordList, offset = (0,0) -> y_values
    generate ycoords from a list of (xcoords, ycoords)'''
    y_values = [coord[1]+offset[1] for coord in coordList]
    return y_values

def get_xdir_list(thetaList, length = 80):
    '''get_xdir_list(thetaList, length = 80) -> xdir_lis
    computes x component based on heading '''
    xdir_list = [length*np.cos(theta*np.pi/180) for theta in thetaList]
    return xdir_list

def get_ydir_list(thetaList, length = 80):
    '''get_ydir_list(thetaList, length = 80) -> ydir_lis
    computes y component based on heading '''
    ydir_list = [length*np.sin(theta*np.pi/180) for theta in thetaList]
    return ydir_list
        

class USLandmark():
    '''Represents a Landmark for US - these are the walls'''

    def __init__(self, lID, startCoords, endCoords, beta):
        self.lID = lID
        self.startCoords = startCoords
        self.endCoords = endCoords
        #if startCoords[0]==endCoords[0]:
        #    self.beta = 90
        #else:
        #    self.beta = np.arctan((endCoords[1]-startCoords[1])/(endCoords[0]-startCoords[0]))*180/np.pi
        #print(self.beta)
        self.beta = beta # slope angle

    def get_beta(self):
        ''' get slope angle'''
        return self.beta

    def get_ID(self):
        ''' get landmark id (wall id)'''
        return self.lID

    def get_coords(self):
        '''start and end coords of a landmark (wall)'''
        return (self.startCoords, self.endCoords)

    def find_distance(self, particle, cwOffset):
        ''' find distance, angle of incidence, and the coords
        at which ultrasonic sensor is aiming at the wall
        See notebook for detailed explaination of how this is done'''
        alpha = particle.get_sensor_heading()
        #print(alpha, self.lID, 'in function')
        ttCoords = particle.get_TT_coords()
        if self.beta == 0: # Wall along x
            deltaY = self.startCoords[1] - ttCoords[1] - cwOffset[1]
            #print('deltaY', deltaY)
            if 0 < alpha < 180 and deltaY > 0:
                #print('here1')
                zL = deltaY/(np.sin(alpha*np.pi/180))
                deltaX = zL*np.cos(alpha*np.pi/180)
                xl = ttCoords[0] + cwOffset[0] + deltaX
                yl = self.startCoords[1] 
                i = 90 - alpha
                if self.startCoords[0] <= xl <= self.endCoords[0] or self.endCoords[0] <= xl <= self.startCoords[0]:
                    return [i, deltaY, zL, (xl,yl)]
                else:
                    return None
            elif 180 < alpha < 360 and deltaY < 0:
                #print('here2')
                zL = deltaY/(np.sin(alpha*np.pi/180))
                deltaX = zL*np.cos(alpha*np.pi/180)
                xl = ttCoords[0] + cwOffset[0] + deltaX
                yl = self.startCoords[1] 
                i = 270-alpha
                #print(i)
                if self.startCoords[0] <= xl <= self.endCoords[0] or self.endCoords[0] <= xl <= self.startCoords[0]:
                 #   print('here2: ',[i, deltaY, zL, (xl,yl)])
                    return [i, deltaY, zL, (xl,yl)]
                else:
                    return None
            else:
                return None
        elif self.beta == 90: # wall along y
            deltaX = self.startCoords[0] - ttCoords[0] - cwOffset[0]
            if 90 < alpha < 270 and deltaX < 0:
                zL = deltaX/(np.cos(alpha*np.pi/180))
                deltaY = zL*np.sin(alpha*np.pi/180)
                xl = self.startCoords[0]
                yl = ttCoords[1] + cwOffset[1] + deltaY
                i = 180 - alpha    
                if self.startCoords[1] <= yl <= self.endCoords[1] or self.endCoords[1] <= yl <= self.startCoords[1]:
                    return [i, deltaX, zL, (xl,yl)]
                else:
                    return None
            elif (alpha < 90 or alpha > 270) and deltaX > 0:
                zL = deltaX/(np.cos(alpha*np.pi/180))
                deltaY = zL*np.sin(alpha*np.pi/180)
                xl = self.startCoords[0]
                yl = ttCoords[1] + cwOffset[1] + deltaY
                i = 360 - alpha
                if self.startCoords[1] <= yl <= self.endCoords[1] or self.endCoords[1] <= yl <= self.startCoords[1]:
                    return [i, deltaX, zL, (xl,yl)]
                else:
                    return None
            else:
                return None
        else: # wall not parallel to the axes
            beta = self.beta
            gamma = alpha - beta
            slope = np.tan(np.pi*beta/180)
            if gamma < 0 or gamma > 180:
                deltaYp = -np.abs(ttCoords[1]+cwOffset[1]-self.startCoords[1] - slope*(ttCoords[0]+cwOffset[0]- self.startCoords[0]))/np.sqrt(1+slope*slope)
                #print('deltaYp ', deltaYp)
                zL = deltaYp/np.sin(gamma*np.pi/180)
                deltaX =  zL*np.cos(alpha*np.pi/180)
                deltaY = zL*np.sin(alpha*np.pi/180)
                if beta == 135:
                    i = 90 + alpha - beta
                else:
                    i = 270 - alpha + beta
                xl = ttCoords[0] + cwOffset[0] + deltaX
                yl = ttCoords[1] + cwOffset[1] + deltaY
                if (self.startCoords[1] <= yl <= self.endCoords[1] or self.endCoords[1] <= yl <= self.startCoords[1]) and (self.startCoords[0] <= xl <= self.endCoords[0] or self.endCoords[0] <= xl <= self.startCoords[0]):
                    return [i, deltaYp, zL, (xl,yl)]
                else:
                    return None
            else:    
                return None
        print('distnace fun should not be here' )


class ColorWorld():
    '''Represents the color world for use with CS - these are the colored squares'''

    def __init__(self, ww):
        self.COLOR_MAP = {
            0: 'None',
            1: 'Color.BLACK',
            2: 'Color.BLUE',
            3: 'Color.GREEN',
            4: 'Color.YELLOW',
            5: 'Color.RED',
            6: 'Color.WHITE',
            7: 'Color.BROWN'
        }
        
        np.random.seed(1234) # set a seed for reproducible randomness
        self.n = 30 # 30 x 30 square color world
        self.ww = ww
        self.squareSize = 40 # 40 mm x 40 mm
        self.colorWorld = np.random.randint(1,8,(self.n,self.n))
        self.whiteCoords = np.stack(np.where(self.colorWorld==6), axis=1)
        self.nTestSites = len(self.whiteCoords)
        self.testCoords = np.stack((np.random.randint(0,315,self.nTestSites),np.random.randint(0,315,self.nTestSites)), axis=1)
        self.testHeadings = np.random.randint(0, 360, self.nTestSites)


    def find_color_true(self, coord):
        '''assumes coords are with respect to the color world bottom right'''
        #print('true color ', coord)
        if -140 <= coord[0] <= 1340 and -140 <= coord[1] <= 1340:
            color = 7
            if -5 <= coord[0] <= 1205 and -5 <= coord[1] <= 1205:
                color = 6
                if 0 <= coord[0] < 1200 and 0 < coord[1] <= 1200:
                    c = int(coord[0] // 40)
                    r = int((1200-coord[1]) // 40)
                    if r >=30 or c >=30:
                        print(c, r, coord)
                    color = self.colorWorld[r,c]
        else:
            print('ERROR in find_color_true',coord)
        return [self.COLOR_MAP[color]]
        

    def find_color(self,coord):
        '''assumes coords are with respect to the color world bottom right'''
        #print(coord)
        if -140 <= coord[0] <= 1340 and -140 <= coord[1] <= 1340:
            color = -1
            if -5 <= coord[0] <= 1205 and -5 <= coord[1] <= 1205:
                color = 6
                if 0 <= coord[0] < 1200 and 0 < coord[1] <= 1200:
                    c = int(coord[0] // 40)
                    r = int((1200-coord[1]) // 40)
                    cRem = coord[0] % 40
                    rRem =  (1200-coord[1]) % 40
                    if cRem < 3 or cRem > 37 or rRem < 3 or rRem > 37:
                        return [self.COLOR_MAP[0], self.COLOR_MAP[self.colorWorld[r,c]]]
                    else:
                        return [self.COLOR_MAP[self.colorWorld[r,c]]]
                if -3 <= coord[0] <= 1203 and -3 <= coord[1] <= 1203:
                    return [self.COLOR_MAP[color]]
        else:
            print(coord)
        if color == -1 or color == 6:
            if coord[0] > -7 or coord[1] > -7 or coord[0] < 1207 or coord[1] < 1207:
                return ['Color.BLACK', 'Color.BROWN', 'Color.WHITE', 'None']
            return ['Color.BLACK', 'Color.BROWN']

    def get_test_coords(self,testSiteID):
        '''This function is in colorWorld Coordinate System'''
        x = self.whiteCoords[testSiteID][1] * 40
        y = 1200-self.whiteCoords[testSiteID][0] * 40
        x = x + self.testCoords[testSiteID][0] * 40/315
        y = y - self.testCoords[testSiteID][1] * 40/315
        return (x,y)

    def get_test_heading(self,testSiteID):
        return self.testHeadings[testSiteID]
    
class WaldoWorld():
    '''Represents Waldo's World - enclosure + color board'''

    def __init__(self):
        '''W()
        initializes the robotMap'''
        coord0 = (0,0)
        coord1 = (0,1509)
        coord2 = (302, 1509)
        coord3 = (302, 1811)
        coord4 = (1516, 1811)
        coord5 = (1816, 1509)
        coord6 = (1816, 604)
        coord7 = (1502, 0)

        self.uslCoords = [coord0,coord1,coord2,coord3,coord4,coord5,coord6,coord7,coord0]
        self.cbCoords = [(266, 187), (266,1397), (1476,1397), (1476, 187), (266, 187)] # Color Board
        self.cwCoords = [(271,191), (271,1391), (1471,1391), (1471,191), (271,191)] # Color World

        self.cwSize = 1200 # square of size 1200 mm x 1200 mm
        self.cbSize = 1210 # square of size 1210 mm x 1210 mm

        self.uslBetaList = [90,0,90,0,135,90,62.53,0]
        self.usLandmarkList = []
        for i in range(8):
            self.usLandmarkList.append(USLandmark(i, self.uslCoords[i], self.uslCoords[i+1],self.uslBetaList[i]))

    def get_cw_size(self):
        return self.cwSize

    def get_cb_xCoords(self):
        return get_xCoords(self.cbCoords)

    def get_cb_yCoords(self):
        return get_yCoords(self.cbCoords)

    def get_cw_xCoords(self):
        return get_xCoords(self.cwCoords)

    def get_cw_yCoords(self):
        return get_yCoords(self.cwCoords)

    def get_cw_offset(self):
        return self.cwCoords[0]

    def get_us_landmarks(self):
        return self.usLandmarkList

    def get_usl_xCoords(self):
        return get_xCoords(self.uslCoords)

    def get_usl_yCoords(self):
        return get_yCoords(self.uslCoords)


        
class Particle():
    '''Represents a Particle'''

    def __init__(self, worldSize, offset = 0, coords = None, theta = None):
        '''Particle()
        initializes a particle
        coords: (x,y) in the colorWorld coordinate system in mm
        theta: robot heading in map coordinate system (wrt x axis in map coords) in degrees
        phi: heading of US/CS wrt to robot coordinate system in degrees
        '''
        
        self.worldSize = worldSize - offset
        if coords == None:
            self.coords = (round(random.random() * self.worldSize + offset/2), round(random.random() * self.worldSize+offset/2))
            self.theta = round(random.random() * 360)
        else:
            self.coords = coords
            self.theta = theta
        
        #self.worldSize = 40
        #self.coords = (713+random.random() * self.worldSize, 658+random.random() * self.worldSize)
        #self.coords = (0+random.random() * self.worldSize, 0+random.random() * self.worldSize)
        #self.theta = random.random() * 360
        
        self.phi = 0
        self.offsetTT = 80 # distance of TT from Robot origin in mm 
        self.offsetUS = 28 # distance of US from TT rotation axis in mm
        self.offsetCS = 60 # distance of CS from TT rotation axis in mm
        self.weight = 1

    def get_coords(self):
        return self.coords

    def get_offsetUS(self):
        return self.offsetUS

    def get_theta(self):
        return self.theta

    def get_phi(self):
        return self.phi

    def set_coords(self,coords):
        self.coords = coords
        return

    def set_theta(self,theta):
        self.theta = theta
        return

    def set_phi(self,phi):
        self.phi = phi
        return

    def get_sensor_heading(self):
        heading = self.theta + self.phi
        if heading >= 360:
            heading -= 360
        if heading < 0:
            heading += 360
        return heading

    def set_weight(self, weight):
        self.weight = weight
        return

    def get_TT_coords(self):
        xTTCoords = self.coords[0] + self.offsetTT * np.cos(self.theta*np.pi/180)
        yTTCoords = self.coords[1] + self.offsetTT * np.sin(self.theta*np.pi/180)
        return (xTTCoords, yTTCoords)

    def get_CS_coords(self):
        '''Particle.get_CS_coords()
        Calculates the coordinates of the Color Sensor in colorWorld Coordinate System'''
        xCSCoord = self.coords[0] + round(self.offsetTT * np.cos(self.theta*np.pi/180) + self.offsetCS * np.cos((self.phi+self.theta)*np.pi/180))
        yCSCoord = self.coords[1] + round(self.offsetTT * np.sin(self.theta*np.pi/180) + self.offsetCS * np.sin((self.phi+self.theta)*np.pi/180))
        return (xCSCoord, yCSCoord)


class ParticleFilter():
    '''Implements a Particle Filter'''

    def __init__(self, nParticles, world, init = 'random'):
        '''ParticleFilter()
        initializes a particle filter by generating particles uniformly on the map
        nParticles: number of particles
        particles are chosen using a uniform random distribution'''
        self.nParticles = nParticles
        self.particleList = []
        self.weightList = np.ones(nParticles)*1.0/nParticles
        self.world = world
        self.usl = self.world.get_us_landmarks()
        self.colorWorld = ColorWorld(self.world)
        worldSize = world.get_cw_size()
        offset = 200
        offsetby2 = int(offset/2)
        if init == 'random':
            #self.particleList = [Particle(worldSize, offset) for i in range(self.nParticles)]
            for i in range(self.nParticles):
                particle = Particle(worldSize, offset)
                #while particle in self.particleList:
                #    particle = Particle(worldSize, offset)
                self.particleList.append(particle)
        else:
            for i in range(offsetby2, worldSize - offsetby2, 10):
                for j in range(offsetby2, worldSize - offsetby2, 10):
                    for k in range(0, 360, 3):
                        particle = Particle(worldSize, offset, (i,j),k)
                        self.particleList.append(particle)
                        
            
        # Process Parameters
        self.phiSigma = 0.0 # degrees
        self.thetaSigma = 0.0 # degrees
        self.posSigma = 0.0 # mm
        self.pHit = 0.8
        self.pMiss = 0.1
        self.pNone = 0.4
        


    def get_particle_list(self):
        return self.particleList
        
    def get_particle_coords_list(self):
        coordList = []
        for particle in self.particleList:
            coordList.append(particle.get_coords())
        return coordList

    def get_particle_theta_list(self):
        thetaList = []
        for particle in self.particleList:
            thetaList.append(particle.get_theta())
        return thetaList

    def get_particle_phi_list(self):
        phiList = []
        for particle in self.particleList:
            phiList.append(particle.get_phi())
        return phiList
    
    def get_weight_list(self):
        return self.weightList
    
    def prediction(self, phi):
        '''ParticleFilter.prediction()
        Predicts the state for the next time stamp using the process model'''
        for particle in self.particleList:

            particle.set_phi(phi)
            #particle.set_theta(theta)
            #particle.set_coords((posX, posY))
            
        return

    def data_association(self):
        '''ParticleFilter.data_association()
        Finds which observations correspond to which landmarks'''
        pass

    def get_unique_particles(self):
        pass

    def update_weights(self, meas1, meas2):
        '''ParticleFilter.update_weights()
        Update the weights for each particle based on the likelihood of the observed measurements'''
        for idx, particle in enumerate(self.particleList):
            color = self.colorWorld.find_color_true(particle.get_CS_coords())
            for j in range(8):
                distance = self.usl[j].find_distance(particle,self.world.get_cw_offset())
                if distance != None:
                    break
            usOffset = particle.get_offsetUS()
            #print(distance)
            #print('idx color meas ', idx, color, meas1)
            if meas1 == 'None':
                self.weightList[idx] *= self.pNone
                particle.set_weight(self.weightList[idx])
            elif meas1 in color:
                self.weightList[idx] *= self.pHit
                particle.set_weight(self.weightList[idx])
            else:
                self.weightList[idx] *= self.pMiss
                particle.set_weight(self.weightList[idx])
        self.weightList = self.weightList/sum(self.weightList) # normalize 
        return 
                
                
    def resample(self):
        '''ParticleFilter.resample()
        Resamples fromm the udpated set of particles to form the new set of particles'''
        if self.nParticles <= 5:
            return
        resampledParticleList = []
        resampledWeightList = []
        index = int(random.random() * self.nParticles)
        beta = 0.0
        mw = max(self.weightList)
        for i in range(self.nParticles):
            beta += random.random() * 2.0 * mw
            while beta > self.weightList[index]:
                beta -=self.weightList[index]
                index = (index + 1) % self.nParticles
            particle = self.particleList[index]
            #resampledParticleList.append(particle)
            #resampledWeightList[i] = self.weightList[index]
            if particle in resampledParticleList:
                resampledWeightList[resampledParticleList.index(particle)] += self.weightList[index]
            else:
                resampledParticleList.append(self.particleList[index])
                resampledWeightList.append(self.weightList[index])
        self.particleList = resampledParticleList
        self.weightList = np.array(resampledWeightList)
        self.weightList = self.weightList/sum(self.weightList) # normalize 
        self.nParticles = len(self.particleList)



def run_particle_filter(nParticles):
    
    
    ww = WaldoWorld()
    cw = ColorWorld(ww)

    testID = 24
    fileStr = "C:/Users/Arnav/OneDrive/Documents/2022 Science Fair Project/Data/Test" + create_3_digit(testID) + ".csv"
    #print(fileStr)
    nLines = 104

    #startTime = time.time()
    pf = ParticleFilter(nParticles,ww)
    #pf = ParticleFilter(1200000,ww,'nr')
    #pL = pf.get_particle_list()

    testCoords = cw.get_test_coords(testID)
    testHeading = cw.get_test_heading(testID)

    #pL[0].set_coords(testCoords)
    #pL[0].set_theta(testHeading)

    cwOffset = ww.get_cw_offset()

    #print("Initialize")

    currentTime = time.time()
    #print('time ', currentTime - startTime) 
    #pCoords = pf.get_particle_coords_list()
    #pThetas = pf.get_particle_theta_list()
    #pWeights = pf.get_weight_list()
    #print(pCoords)
    #print(pWeights)


    df1 = pd.read_csv(fileStr)
    for i in range(0,nLines,4):
        #print("Processing meas " + str(i))
        phi = df1[' tt_angle'][i] * 180/205 + 90
        meas1 = df1[' color'][i].strip()
        meas2 = df1[' distance'][i]
        pf.prediction(phi)
        pf.update_weights(meas1, meas2)
        #print(phi)
        #print('post update wts')
        #pCoords = pf.get_particle_coords_list()
        #pThetas = pf.get_particle_theta_list()
        #pWeights = pf.get_weight_list()
        #print(pCoords)
        #print(pWeights)
        #print(pThetas)
        pf.resample()
        #print('post resample')
        #pCoords = pf.get_particle_coords_list()
        #pThetas = pf.get_particle_theta_list()
        #pWeights = pf.get_weight_list()
        #print(pCoords)
        #print(pWeights)
        #print(pThetas)
    

    '''
    fig, ax = plt.subplots()
    ax.set_title('Localization in Waldo World')
    ax.set_xlabel('mm')
    ax.set_ylabel('mm')
    ax.scatter(testCoords[0]+cwOffset[0],testCoords[1]+cwOffset[1])
    ax.plot(ww.get_usl_xCoords(),ww.get_usl_yCoords())
    ax.plot(ww.get_cb_xCoords(), ww.get_cb_yCoords())
    ax.plot(ww.get_cw_xCoords(), ww.get_cw_yCoords())

    ax.scatter(get_xCoords(pCoords,cwOffset), get_yCoords(pCoords,cwOffset), s = 100/max(pWeights)*pWeights, alpha = 0.5)
    ax.set_aspect('equal')
    ax.set_xlim(right = 2000)
    ax.set_ylim(top = 2000)
    plt.show(block = False)
    '''

#eval_CS_perf_strips()
eval_CS_perf(36)
#eval_US_perf(36)

    
'''
particles = [100, 1000, 10000, 100000, 1000000]
#particles = [100]
for p in particles:
    startTime = time.time()
    run_particle_filter(p)
    endTime = time.time()
    print(p, endTime-startTime) 
'''  

'''
def eval_vectorize_perf():
    testID = 24
    nPL = [100, 1000, 10000, 100000, 1000000]
    logFile = 'vec_perf.csv'
    outFile = open(logFile,'w')
    headerStr = 'testSite, nParticles, nMeasurements, time(s) \n'
    outFile.write(headerStr)
    for nParticles in nPL:
        startTime = time.time()
        pDF = run_particle_filter(testID, nParticles, step)
        endTime = time.time()
        tString = '{:3.6f}'.format(endTime-startTime)
        logString = '{}, {}, {}, {}\n'.format(testID, nParticles, 26,tString)  
        outFile.write(logString)
    outFile.close()
'''
