'''ParticleFilterImproved Module
Implements the following with Vectorized Operations (minimal lists, for loops):
1. Partcile filter, ColorWorld and WaldoWorld classes
2. functions to execute and evaluate the performance of a particle filtera Particle Filter, functions to evaluate '''

import random
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.image as mpimg
import cv2 as cv
from scipy.stats import norm

    
random.seed(2222022) # set the seed so that the color world random numbers are same run to run
offsetTT = 80 # distance of TT from Robot origin in mm 
offsetUS = 28 # distance of US from TT rotation axis in mm
offsetCS = 60 # distance of CS from TT rotation axis in mm
#dictionary for lego color map
colorMap = {
            'None': 0,
            'Color.BLACK': 1,
            'Color.BLUE': 2,
            'Color.GREEN': 3,
            'Color.YELLOW': 4,
            'Color.RED': 5,
            'Color.WHITE': 6,
            'Color.BROWN': 7
        }
        

def evaluate_weighted_performance(pDF, gCoord, gHeading):
    '''evaluate_weighted_performance(pDF, gCoord, gHeading) -> weighted_position_error, weighted_heading_error
    Compute weighted position and heading error of the particles in the data frame wrt ground truth'''
    position_error = np.sqrt((pDF['x']-gCoord[0])**2 + (pDF['y']-gCoord[1])**2)
    heading_error = np.absolute(pDF['theta']-gHeading)
    weighted_position_error = np.sum(position_error*pDF['weight'])/np.sum(pDF['weight'])
    weighted_heading_error = np.sum(heading_error*pDF['weight'])/np.sum(pDF['weight'])
    return weighted_position_error, weighted_heading_error


def evaluate_best_performance(pDF, gCoord, gHeading):
    '''evaluate_best_performance(pDF, gCoord, gHeading) -> position_error, heading_error
    Compute position and heading error wrt ground truth of the partile with maximum weight'''
    max_weight = pDF['weight'].max()
    idx = np.where(pDF['weight']== max_weight)[0]
    pDFBest = pDF.loc[idx]
    position_error = np.sqrt((pDFBest['x']-gCoord[0])**2 + (pDFBest['y']-gCoord[1])**2)
    heading_error = np.absolute(pDFBest['theta']-gHeading)
    idx = np.argmin(position_error) # in case there are multiple with same weight, find the minimum error one
    return position_error[position_error.index[idx]], heading_error[heading_error.index[idx]]

    
def create_3_digit(num):
    '''create_3_digit(num) -> 3 digit string
    Assumes given numbers have 3 or less digits'''
    num = str(num)
    if len(num) == 3:
        return str(num)
    if len(num) == 2:
        return '0' + num
    if len(num) == 1:
        return '00' + num
    

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
        self.beta = beta

    def get_beta(self):
        ''' get slope angle'''
        return self.beta

    def get_ID(self):
        ''' get landmark id (wall id)'''
        return self.lID

    def get_coords(self):
        '''start and end coords of a landmark (wall)'''
        return (self.startCoords, self.endCoords)

    def find_distance(self, nParticles, pXCoordList, pYCoordList, pThetaList, pPhiList, cwOffset, normDistList, iAngleList):
        ''' find normal distance, angle of incidence at the wall
        to which the ultrasonic sensor is aiming at
        See notebook for detailed explaination of how this is done'''
        pAlphaList = pThetaList + pPhiList
        pAlphaList[pAlphaList >=360] -= 360
        pAlphaList[pAlphaList<0] += 360

        xTTCoordList = pXCoordList + offsetTT * np.cos(pThetaList*np.pi/180)
        yTTCoordList = pYCoordList + offsetTT * np.sin(pThetaList*np.pi/180)

        if self.beta == 0: # 0 Slope, parallel to x axis
            deltaYList = self.startCoords[1] - yTTCoordList - cwOffset[1]
            # Condition for hitting walls parallel to x axis
            idx1 = (0 < pAlphaList) & (pAlphaList < 180) & (deltaYList > 0)
            zL = deltaYList[idx1]/(np.sin(pAlphaList[idx1]*np.pi/180))
            deltaX = zL*np.cos(pAlphaList[idx1]*np.pi/180)
            xl = xTTCoordList[idx1] + cwOffset[0] + deltaX
            # Condition for hitting the wall within its span
            idx2 = ((self.startCoords[0] <= xl) & (xl <= self.endCoords[0]))|((self.endCoords[0] <= xl) & (xl <= self.startCoords[0]))
            idx = np.where(idx1==True)[0][idx2]
            normDistList[idx] = np.absolute(deltaYList[idx])
            iAngleList[idx] = pAlphaList[idx] - 90
            # Condition for hitting the wall within its span
            idx1 = (180 < pAlphaList) & (pAlphaList < 360) & (deltaYList < 0)
            zL = deltaYList[idx1]/(np.sin(pAlphaList[idx1]*np.pi/180))
            deltaX = zL*np.cos(pAlphaList[idx1]*np.pi/180)
            xl = xTTCoordList[idx1] + cwOffset[0] + deltaX
            idx2 = ((self.startCoords[0] <= xl) & (xl <= self.endCoords[0]))|((self.endCoords[0] <= xl) & (xl <= self.startCoords[0]))
            idx = np.where(idx1==True)[0][idx2]
            normDistList[idx] = np.absolute(deltaYList[idx])          
            iAngleList[idx] = pAlphaList[idx] - 270
        elif self.beta == 90: # parallel to y axis
            deltaXList = self.startCoords[0] - xTTCoordList - cwOffset[0]
            idx1 = (90 < pAlphaList) & (pAlphaList < 270) & (deltaXList < 0)
            zL = deltaXList[idx1]/(np.cos(pAlphaList[idx1]*np.pi/180))
            deltaY = zL*np.sin(pAlphaList[idx1]*np.pi/180)
            yl = yTTCoordList[idx1] + cwOffset[1] + deltaY
            idx2 = ((self.startCoords[1] <= yl) & (yl <= self.endCoords[1]))|((self.endCoords[1] <= yl) & (yl <= self.startCoords[1]))
            idx = np.where(idx1==True)[0][idx2]
            normDistList[idx] = np.absolute(deltaXList[idx])
            iAngleList[idx] = pAlphaList[idx] - 180
            idx1 = (pAlphaList < 90) & (deltaXList > 0)
            zL = deltaXList[idx1]/(np.cos(pAlphaList[idx1]*np.pi/180))
            deltaY = zL*np.sin(pAlphaList[idx1]*np.pi/180)
            yl = yTTCoordList[idx1] + cwOffset[1] + deltaY
            idx2 = ((self.startCoords[1] <= yl) & (yl <= self.endCoords[1]))|((self.endCoords[1] <= yl) & (yl <= self.startCoords[1]))
            idx = np.where(idx1==True)[0][idx2]
            normDistList[idx] = np.absolute(deltaXList[idx])
            iAngleList[idx] = pAlphaList[idx]
            idx1 = (pAlphaList > 270) & (deltaXList > 0)
            zL = deltaXList[idx1]/(np.cos(pAlphaList[idx1]*np.pi/180))
            deltaY = zL*np.sin(pAlphaList[idx1]*np.pi/180)
            yl = yTTCoordList[idx1] + cwOffset[1] + deltaY
            idx2 = ((self.startCoords[1] <= yl) & (yl <= self.endCoords[1]))|((self.endCoords[1] <= yl) & (yl <= self.startCoords[1]))
            idx = np.where(idx1==True)[0][idx2]
            normDistList[idx] = np.absolute(deltaXList[idx])
            iAngleList[idx] = pAlphaList[idx] - 360
        else: # wall not parallel to the axes
            gammaList = pAlphaList - self.beta
            slope = np.tan(np.pi*self.beta/180)
            deltaYp = -np.abs(yTTCoordList+cwOffset[1]-self.startCoords[1] - slope*(xTTCoordList+cwOffset[0]- self.startCoords[0]))/np.sqrt(1+slope*slope)
            idx1 = (gammaList < 0)
            zL = deltaYp[idx1]/np.sin(gammaList[idx1]*np.pi/180)
            deltaX =  zL*np.cos(pAlphaList[idx1]*np.pi/180)
            deltaY = zL*np.sin(pAlphaList[idx1]*np.pi/180)
            xl = xTTCoordList[idx1] + cwOffset[0] + deltaX
            yl = yTTCoordList[idx1] + cwOffset[1] + deltaY
            idx2 = ((((self.startCoords[1] <= yl) & (yl <= self.endCoords[1]))
                    |((self.endCoords[1] <= yl) & (yl <= self.startCoords[1])))
                    &(((self.startCoords[0] <= xl) & (xl <= self.endCoords[0]))
                    |((self.endCoords[0] <= xl)&(xl <= self.startCoords[0]))))
            idx = np.where(idx1==True)[0][idx2]
            normDistList[idx] = np.absolute(deltaYp[idx])
            iAngleList[idx] = 90 + pAlphaList[idx] - self.beta
            idx1 = (gammaList > 180)
            zL = deltaYp[idx1]/np.sin(gammaList[idx1]*np.pi/180)
            deltaX =  zL*np.cos(pAlphaList[idx1]*np.pi/180)
            deltaY = zL*np.sin(pAlphaList[idx1]*np.pi/180)
            xl = xTTCoordList[idx1] + cwOffset[0] + deltaX
            yl = yTTCoordList[idx1] + cwOffset[1] + deltaY
            idx2 = ((((self.startCoords[1] <= yl) & (yl <= self.endCoords[1]))
                    |((self.endCoords[1] <= yl) & (yl <= self.startCoords[1])))
                    &(((self.startCoords[0] <= xl) & (xl <= self.endCoords[0]))
                    |((self.endCoords[0] <= xl)&(xl <= self.startCoords[0]))))
            idx = np.where(idx1==True)[0][idx2]
            normDistList[idx] = np.absolute(deltaYp[idx])
            iAngleList[idx] = pAlphaList[idx] - self.beta - 270



class ColorWorld():
    '''Represents the color world for use with CS - these are the colored squares'''

    def __init__(self, ww):
        np.random.seed(1234) # set a seed for reproducible randomness
        self.n = 30 # 30 x 30 square color world
        self.ww = ww
        self.squareSize = 40 # 40 mm x 40 mm
        self.colorWorld = np.random.randint(1,8,(self.n,self.n))
        self.whiteCoords = np.stack(np.where(self.colorWorld==6), axis=1)
        self.nTestSites = len(self.whiteCoords)
        self.testCoords = np.stack((np.random.randint(0,315,self.nTestSites),np.random.randint(0,315,self.nTestSites)), axis=1)
        self.testHeadings = np.random.randint(0, 360, self.nTestSites)


    def find_color_true(self, nParticles, cXCoordList, cYCoordList):
        ''' Finds true colors from image of a list of x and y coords
        representing color sensor coords
        assumes coords are with respect to the color world bottom right'''
        colorList = np.full(nParticles,7)
        bArr =  (-5 <= cXCoordList) &(cXCoordList <= 1205)& (-5 <= cYCoordList) & (cYCoordList <= 1205)
        colorList[bArr] = 6
        bArr = (0 <= cXCoordList) & (cXCoordList < 1200) & (0 < cYCoordList) & (cYCoordList <= 1200)
        colorList[bArr] = self.colorWorld[((1200-cYCoordList[bArr])//40).astype(int),(cXCoordList[bArr]//40).astype(int)]
        return colorList

    def get_test_coords(self,testSiteID):
        '''Find test coords of a testSite
        This function is in colorWorld Coordinate System'''
        x = self.whiteCoords[testSiteID][1] * 40
        y = 1200-self.whiteCoords[testSiteID][0] * 40
        x = x + self.testCoords[testSiteID][0] * 40/315
        y = y - self.testCoords[testSiteID][1] * 40/315
        return (x,y)

    def get_test_heading(self,testSiteID):
        '''Heading at the test site'''
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



class ParticleFilter():
    '''Implements a Particle Filter'''

    def __init__(self, nParticles, world, init = 'random'):
        '''ParticleFilter()
        initializes a particle filter by generating particles uniformly on the map
        nParticles: number of particles
        particles are chosen using a uniform random distribution'''
        self.nParticles = nParticles
        self.weightList = np.ones(nParticles)
        self.pXCoordList = np.zeros(nParticles)
        self.pYCoordList = np.zeros(nParticles)
        self.pThetaList = np.zeros(nParticles)
        self.pPhiList = np.zeros(nParticles)

        self.world = world
        self.usl = self.world.get_us_landmarks()
        self.colorWorld = ColorWorld(self.world)
        worldSize = world.get_cw_size()
        offset = 200
        offsetby2 = int(offset/2)
        if init == 'random':
            self.pXCoordList = np.random.uniform(offsetby2, worldSize - offsetby2, nParticles)
            self.pYCoordList = np.random.uniform(offsetby2, worldSize - offsetby2, nParticles)
            self.pThetaList = np.random.uniform(0,360, nParticles)
        else:
            xyPts = (worldSize - offset)//10
            thetaPts = 120
            xyArr = np.arange(offsetby2, worldSize - offsetby2, 10)
            thetaArr = np.arange(0,360,3)
            self.pXCoordList = np.repeat(xyArr, xyPts*thetaPts)
            self.pYCoordList = np.tile(np.repeat(xyArr, thetaPts), xyPts)
            self.thetaList = np.tile(thetaArr, xyPts*xyPts)
            
        # Process Parameters
        self.phiSigma = 0.25 # degrees
        self.thetaSigma = 0 # degrees
        self.posSigma = 0 # mm
        
        # Probaility models for weights
        #Color Sensor probs
        self.pHit = 0.80
        self.pMiss = 0.20
        self.pNone = 0.1
        # Ultrasonic sensor scaling factors and sigma
        self.usPScale = 2.25
        self.usNScale = 1.8
        self.usScale1 = 0.0001
        self.usSigma = 20
        #self.usScale1 = 1e-2*norm.pdf(0,0,self.usSigma)
        #self.usScale2 = 1e-5*norm.pdf(0,0,self.usSigma)
        
        #self.usSigma = 0.5*worldSize/np.sqrt(nParticles)
        #print('usSigma ', self.usSigma)
        #if self.usSigma < 5:
        #    self.usSigma = 5

        
        
    def set_pCoord(self, idx, coord, theta):
        self.pXCoordList[idx] = coord[0]
        self.pYCoordList[idx] = coord[1]
        self.pThetaList[idx] = theta

    def get_pXCoordList(self):
        return self.pXCoordList

    def get_pYCoordList(self):
        return self.pYCoordList

    def get_pThetaList(self):
        return self.pThetaList

    def get_pPhiList(self):
        return self.pPhiList
    
    def get_weight_list(self):
        return self.weightList

    def get_pData(self):
        pData = np.column_stack((self.pXCoordList, self.pYCoordList, self.pThetaList, self.weightList))
        df = pd.DataFrame(data = pData, index = np.arange(self.nParticles), columns = ['x','y','theta','weight'])
        return df
    
    
    def prediction(self, phi):
        '''ParticleFilter.prediction()
        Predicts the state for the next time stamp using the process model'''
        self.pXCoordList = np.random.normal(self.pXCoordList, self.posSigma)
        self.pYCoordList = np.random.normal(self.pYCoordList, self.posSigma)
        self.pThetaList = np.random.normal(self.pThetaList, self.thetaSigma)
        self.pPhiList.fill(phi)
        self.pPhiList = np.random.normal(self.pPhiList, self.phiSigma)
        return

    def data_association(self):
        '''ParticleFilter.data_association()
        Finds which observations correspond to which landmarks'''
        pass


    def update_weights_cs(self, meas1):
        '''ParticleFilter.update_weights_cs()
        Update the weights for each particle based on the likelihood of the observed color measurements'''

        cXCoordList = self.pXCoordList + np.round(offsetTT * np.cos(self.pThetaList*np.pi/180) + offsetCS * np.cos((self.pPhiList+self.pThetaList)*np.pi/180))
        cYCoordList = self.pYCoordList + np.round(offsetTT * np.sin(self.pThetaList*np.pi/180) + offsetCS * np.sin((self.pPhiList+self.pThetaList)*np.pi/180))
        colorList = self.colorWorld.find_color_true(self.nParticles,cXCoordList,cYCoordList)

        hitIndices = colorList == meas1
        missIndices = colorList != meas1
        self.weightList[hitIndices] *= self.pHit
        self.weightList[missIndices] *= self.pMiss
        self.weightList = self.weightList/np.sum(self.weightList) # normalize 
        return 
                                                                                          
    def update_weights_us(self, meas2):
        '''ParticleFilter.update_weights_us()
        Update the weights for each particle based on the likelihood of the observed ultrasonic sensor measurements'''
        if meas2 == 2550:
            return
        normDistList = np.zeros(self.nParticles)
        iAngleList = np.zeros(self.nParticles)
        iAngleList.fill(90)
        prob = np.ones(self.nParticles)
        cwOffset = self.world.get_cw_offset()
        for j in range(8):
            self.usl[j].find_distance(self.nParticles, self.pXCoordList, self.pYCoordList, self.pThetaList, self.pPhiList, cwOffset, normDistList, iAngleList)
        idx = iAngleList >= 0
        prob[idx]= norm.pdf(meas2+offsetUS, normDistList[idx]/np.cos(iAngleList[idx]*np.pi/(180*self.usPScale)), self.usSigma)
        idx = iAngleList < 0
        prob[idx] = norm.pdf(meas2+offsetUS, normDistList[idx]/np.cos(iAngleList[idx]*np.pi/(180*self.usNScale)), self.usSigma)
        #if meas2 == 2550:
        #    idx = np.abs(iAngleList) > 45
        #    prob[idx] = self.usScale1
            #idx = np.abs(iAngleList) <= 45
            #prob[idx] = self.usScale2
        
        self.weightList *= prob
        #if np.sum(self.weightList) == 0:
        if np.max(self.weightList) < 1e-100:
            self.weightList.fill(1)
        #print('sum ',np.max(self.weightList))
        self.weightList = self.weightList/np.sum(self.weightList) # normalize
        return 
        
                
    def resample(self):
        '''ParticleFilter.resample()
        Resamples fromm the udpated set of particles to form the new set of particles'''
        #self.weightList = self.weightList/np.sum(self.weightList) #normalize weightList
        resampledIndices = np.random.choice(self.nParticles, self.nParticles, p = self.weightList)
        
        #resampledPXCoordList = self.pXCoordList[resampledIndices]
        #resampledPYCoordList = self.pYCoordList[resampledIndices]
        #resampledthetaList = self.pThetaList[resampledIndices]
        #resampledphiList = self.pPhiList[resampledIndices]
        #resampledWeightList = self.weightList[resampledIndices]
        
        (uniqueIndices, counts) = np.unique(resampledIndices,return_counts = True)
        #print(uniqueIndices, counts)
        #if uniqueIndices.size < 50:
        #    return
        resampledPXCoordList = self.pXCoordList[uniqueIndices]
        resampledPYCoordList = self.pYCoordList[uniqueIndices]
        resampledthetaList = self.pThetaList[uniqueIndices]
        resampledphiList = self.pPhiList[uniqueIndices]
        resampledWeightList = self.weightList[uniqueIndices]*counts
        self.nParticles = uniqueIndices.size
        #print(resampledWeightList)
        
        self.pXCoordList = resampledPXCoordList
        self.pYCoordList = resampledPYCoordList
        self.pThetaList = resampledthetaList
        self.pPhiList = resampledphiList
        self.weightList = resampledWeightList/np.sum(resampledWeightList)
        return


def plot_localization_results(pDF, testCoords, ww, titleString = 'Localization in Waldo World'):
    pXCoordList = pDF['x']
    pYCoordList = pDF['y']
    pThetaList = pDF['theta']
    cwOffset = ww.get_cw_offset()
    pWeights = pDF['weight']

    fig, ax = plt.subplots()
    ax.set_title(titleString, fontsize = 10)
    ax.set_xlabel('mm')
    ax.set_ylabel('mm')
    ax.scatter(testCoords[0]+cwOffset[0],testCoords[1]+cwOffset[1])
    ax.plot(ww.get_usl_xCoords(),ww.get_usl_yCoords())
    ax.plot(ww.get_cb_xCoords(), ww.get_cb_yCoords())
    ax.plot(ww.get_cw_xCoords(), ww.get_cw_yCoords())

    ax.scatter(pXCoordList+cwOffset[0], pYCoordList+cwOffset[1], s = 100/max(pWeights)*pWeights, alpha = 0.5)
    #ax.scatter(pXCoordList+cwOffset[0], pYCoordList+cwOffset[1])
    ax.set_aspect('equal')
    ax.set_xlim(right = 2000)
    ax.set_ylim(top = 2000)
    plt.show(block = False)

    
def run_particle_filter(testID, nParticles, nLines, step, sensor, init, genPlot = False):
    
    ww = WaldoWorld()
    cw = ColorWorld(ww)

    fileStr = "C:/Users/Arnav/OneDrive/Documents/2022 Science Fair Project/Data/Test" + create_3_digit(testID) + ".csv"
    testCoords = cw.get_test_coords(testID)
    testHeading = cw.get_test_heading(testID)
    
    startTime = time.time()
    if init == 'ru':
        pf = ParticleFilter(nParticles,ww)
    else:
        pf = ParticleFilter(nParticles,ww,'nr')
    #pf = ParticleFilter(1200000,ww)
    #pf = ParticleFilter(1200000,ww,'nr')
    #pf.set_pCoord(0, testCoords, testHeading)

    df1 = pd.read_csv(fileStr)
    count = 0
    for i in range(0,nLines,step):
        phi = df1[' tt_angle'][i] * 180/205 + 90
        meas1 = colorMap[df1[' color'][i].strip()]
        meas2 = df1[' distance'][i]
        pf.prediction(phi)
        
        if sensor == 'cs':
            pf.update_weights_cs(meas1)
        elif sensor == 'us': 
            pf.update_weights_us(meas2)
        else:
            pf.update_weights_cs(meas1)
            pf.update_weights_us(meas2)
            
        if i>0 and i%10 == 0:
            pf.resample()
        
        pDF = pf.get_pData()
        count+=1
        titleStr = 'State After Processing Color Sensor Measurement '+ str(count)
        #titleStr = 'Initial State: 100,000 Uniformly Distributed Particles'
        plot_localization_results(pDF, testCoords, ww,titleStr)
        #figStr = 'US'+str(i)+'.jpg'
        figStr = 'L' + str(testID)+ sensor + str(i) + '.jpg'
        #figStr = 'Linit.jpg'
        plt.savefig(figStr) 

    return
    '''
    endTime = time.time()
    pDF = pf.get_pData()
    wPosError, wHeadingError = evaluate_weighted_performance(pDF, testCoords, testHeading)
    bPosError, bHeadingError = evaluate_best_performance(pDF, testCoords, testHeading)
    tString = '{:3.6f}'.format(endTime-startTime)
    wPosErrString = '{:4.0f}'.format(wPosError)
    wHeadingErrString = '{:3.0f}'.format(wHeadingError)
    bPosErrString = '{:4.0f}'.format(bPosError)
    bHeadingErrString = '{:3.0f}'.format(bHeadingError)
    fPE = np.sqrt((testCoords[0]-600)**2+(testCoords[1]-600)**2)
    xR = np.random.uniform(100, 1100, 1)
    yR = np.random.uniform(100, 1100, 1)
    rPE = np.sqrt((testCoords[0]-xR)**2+(testCoords[1]-yR)**2)
    fPEString = '{:4.0f}'.format(fPE)
    rPEString = '{:4.0f}'.format(rPE[0])
    logString = '{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(testID, nParticles, 26, sensor, init, \
                    wPosErrString, wHeadingErrString, bPosErrString, bHeadingErrString, fPEString, rPEString, tString)
    if genPlot == True:
        plot_localization_results(pDF, testCoords, ww)
    return logString
    '''



def eval_particle_filter():
    '''eval_particle_filter()
        Goes through all the test sites and evaluates particle filter performance
        for different number of particles in terms of time and best and weighted errors
        Results are logged in a file - one entry per test site, per nParticles, per sensor
        Plot results optionally'''
    
    # setup paramaters and logging
    ww = WaldoWorld()
    cw = ColorWorld(ww)
    nLines = 104
    step = 4

    nPL = [100, 1000, 10000, 100000, 1000000]
    #nPL = [10000]
    testSites = np.arange(6,126)
    #testSites = [36]
    logFile = 'localization_perf.csv'
    outFile = open(logFile,'w')
    headerStr = 'testSite,nParticles,nMeas,sensor,init,wPE(mm),wHE(degree),bPE(mm),bHE(degree),fPE(mm),rPE(mm),time(s)\n'
    outFile.write(headerStr)

    for testID in testSites: #go through each test site
        print('processing test site ', testID)
        testCoords = cw.get_test_coords(testID)
        testHeading = cw.get_test_heading(testID)
        # Dont include edge test sites as color sensor measurements not available beyond color world 
        if 100<testCoords[0]<1100 and 100<testCoords[1]<1100: 
            for nParticles in nPL:
                logString = run_particle_filter(testID, nParticles, nLines, step, 'cs', 'ru')
                outFile.write(logString)
                print(logString)
                logString = run_particle_filter(testID, nParticles, nLines, step, 'us', 'ru')   
                outFile.write(logString)
                print(logString)
                if nParticles == 1200000:
                    logString = run_particle_filter(testID, nParticles, nLines, step, 'cs', 'fu')
                    outFile.write(logString)
                    print(logString)
                    logString = run_particle_filter(testID, nParticles, nLines, step, 'us', 'fu')   
                    outFile.write(logString)
                    print(logString)
                logString = run_particle_filter(testID, nParticles, nLines, step, 'fs', 'ru')   
                outFile.write(logString)
                print(logString)    
                #plot_localization_results(pDF, testCoords, ww)
                
    outFile.close()
        


#eval_particle_filter()
run_particle_filter(66, 1000000, 104, 4, 'cs', 'ru')

    
    
