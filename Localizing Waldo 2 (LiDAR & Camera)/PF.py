import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.image as mpimg
import WaldosWorld
import GTWheelOdometry
from scipy.stats import norm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

np.random.seed(20230101) # set the seed so that random numbers are same run to run

def load_RPLidar_data(fileStr):
    data = np.load(fileStr, allow_pickle = True)
    nScans = len(data)
    nSamples = 0
    for i in range(nScans):
        nSamples += len(data[i][3])
    dist = np.array([None]*nSamples)
    azimuth = np.array([None]*nSamples)
    t = np.array([None]*nSamples)
    start_idx = 0
    st = data[0][1]
    for i in range(nScans):
        #st = data[i][1]
        et = data[i][2]
        n = len(data[i][3])
        #print(st,et,n)
        abs_t = np.linspace(st,et,n)
        d = data[i][3][:,2]/10
        az = (180-data[i][3][:,1] + 360)%360
        az = az*np.pi/180
        dist[start_idx:start_idx+n] = d
        azimuth[start_idx:start_idx+n] = az
        t[start_idx:start_idx+n] = np.round(abs_t, 1)
        start_idx += n
        st = et
    return t, dist, azimuth

def get_RPLidar_data(t, dist, azimuth, abs_t):
    idx = t == abs_t
    return dist[idx], azimuth[idx]

def load_TFLuna(fileStr):
    data = np.load(fileStr, allow_pickle = True)
    start = data[0]
    end = data[1]
    scan = np.array(data[2])
    d = scan[:,2]*100
    az = (scan[:,1]-start)*np.pi*2/(end-start)
    return d, az


class LiDARLandmark():
    '''Represents a Landmark for LiDAR - these are the walls'''
    def __init__(self, lID, startCoords, endCoords, beta, coeff):
        self.lID = lID
        self.startCoords = startCoords
        self.endCoords = endCoords
        self.beta = beta
        self.coeff = coeff

    def get_beta(self):
        ''' get slope angle'''
        return self.beta

    def get_ID(self):
        ''' get landmark id (wall id)'''
        return self.lID

    def get_coords(self):
        '''start and end coords of a landmark (wall)'''
        return (self.startCoords, self.endCoords)

    def find_distance(self, x, y, theta, dist):
        '''Find distance from the landmark (wall)'''
        # Theta is in radians, function epxects in degrees
        theta = (theta * 180/np.pi) % 360
        xStartCoords = self.startCoords[0]-x
        yStartCoords = self.startCoords[1]-y
        xEndCoords = self.endCoords[0]-x
        yEndCoords = self.endCoords[1]-y
        angles1 = ((np.arctan2(yStartCoords, xStartCoords) * 180 / np.pi)+360)%360
        angles2 = ((np.arctan2(yEndCoords, xEndCoords) * 180 / np.pi)+360)%360
        idx1 = np.abs(angles1-angles2) >= 180
        if idx1.size > 0:
            big = np.maximum(angles1[idx1], angles2[idx1])
            small = np.minimum(angles1[idx1], angles2[idx1])
            idx2 = ((0 <= theta[idx1]) & (theta[idx1] <= small)) | ((big <= theta[idx1]) & (theta[idx1]<= 360))
            if idx2.size > 0:
                idx = np.where(idx1==True)[0][idx2]
                if idx.size == 0:
                    return
                coeff = np.tan(theta[idx]/180*np.pi)
                a1 = np.stack((coeff,-1*np.ones_like(coeff)), axis = -1)
                a2 = np.stack((self.beta*np.ones_like(coeff), -1*np.ones_like(coeff)), axis = -1)
                a = np.stack((a2, a1), axis = 1)
                b = np.stack((-yStartCoords[idx]+self.beta*xStartCoords[idx], np.zeros_like(coeff)), axis = -1)
                z = np.linalg.solve(a, b)
                dist[idx] = np.sqrt(np.sum(z*z,axis=1))
        
        idx1 = np.abs(angles1-angles2) < 180
        if idx1.size > 0:
            idx2 = ((angles1[idx1] <= theta[idx1])&(theta[idx1] <= angles2[idx1])) | ((angles2[idx1] <= theta[idx1]) & (theta[idx1] <= angles1[idx1]))
            if idx2.size > 0:
                idx = np.where(idx1==True)[0][idx2]
                if idx.size == 0:
                    return
                coeff = np.tan(theta[idx]/180*np.pi)
                a1 = np.stack((coeff,-1*np.ones_like(coeff)), axis = -1)
                a2 = np.stack((self.beta*np.ones_like(coeff), -1*np.ones_like(coeff)), axis = -1)
                a = np.stack((a2, a1), axis = 1)
                b = np.stack((-yStartCoords[idx]+self.beta*xStartCoords[idx], np.zeros_like(coeff)), axis = -1)
                z = np.linalg.solve(a, b)
                dist[idx] = np.sqrt(np.sum(z*z,axis=1))

class WaldoRaceTrack():
    '''Represents Waldo's Racetrack - enclosure + tracks + waypoints'''

    def __init__(self):
        '''WaldoRaceTrack()
        initializes the robotMap'''

        self.wallCoords = WaldosWorld.wallCoords
        self.wayPoints = WaldosWorld.wpCoords
        self.wallBetaArr = WaldosWorld.wallBetaArr
        self.wallCoeffArr = WaldosWorld.wallCoeffArr
        self.lidarLandmarkList = []

        for i in range(7):
            self.lidarLandmarkList.append(LiDARLandmark(i, self.wallCoords[i], self.wallCoords[i+1],self.wallBetaArr[i],self.wallCoeffArr[i]))

    def get_lidar_landmarks(self):
        return self.lidarLandmarkList

    def get_wall_coords(self):
        return self.wallCoords

    def get_wall_beta(self):
        return self.wallBetaArr

    def get_wall_coeff(self):
        return self.wallCoeffArr

class ParticleFilter():
    '''Implements a Particle Filter'''

    def __init__(self, nParticles, wrt, initState, initSigma):
        '''ParticleFilter()
        initializes a particle filter by generating particles uniformly on the map
        nParticles: number of particles
        particles are chosen using a normal random distribution
        Assumption: particles are representing body frame origin'''
        self.nParticles = nParticles
        self.weightList = np.ones(nParticles)

        self.pPhiList = np.zeros(nParticles)# azimuth of LiDAR

        # Normal Distribution around initial state
        self.pXCoordList = np.random.normal(initState[0], initSigma[0], nParticles)
        self.pYCoordList = np.random.normal(initState[1], initSigma[1], nParticles)
        self.pThetaList = np.random.normal(initState[2], initSigma[2], nParticles)
        #Everything should be within the enclosure
        idx = self.pXCoordList < 30.0
        self.pXCoordList[idx] = 30.0
        idx = self.pXCoordList > 260.0
        self.pXCoordList[idx] = 260.0
        idx = self.pYCoordList < 30.0
        self.pYCoordList[idx] = 30.0
        idx = self.pYCoordList > 260.0
        self.pYCoordList[idx] = 260.0
        
        self.world = wrt
            
        # Process Noise
        #self.sigma = np.array([0.5, 0.5, 0.005]) # in cm, cm, radians
        self.sigma = np.array([0., 0., 0.01]) # in cm, cm, radians
        self.sigmaLandmark = 5 #cm
        #self.sigmaLandmark = 0.0 #cm

    def set_pCoord(self, idx, state):
        self.pXCoordList[idx] = state[0]
        self.pYCoordList[idx] = state[1]
        self.pThetaList[idx] = state[2]

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
    
    
    def prediction(self, wL, wR):
        '''ParticleFilter.prediction()
        Predicts the state for the next time stamp using the kinematic model '''
        #Hold the state as is if robot did not move
        if wL==0.0 and wR==0.0:
            print("Static lockdown")
            return
        #First move the particles to Axel Frame 
        self.pXCoordList, self.pYCoordList = WaldosWorld.transform(self.pXCoordList, self.pYCoordList, self.pThetaList,
                                                       WaldosWorld.axleCoords, WaldosWorld.bodyCoords)
        #Apply kinematic model
        self.pXCoordList = self.pXCoordList + WaldosWorld.r/2 * (-wL+wR)*np.cos(self.pThetaList)*WaldosWorld.deltaT
        self.pYCoordList = self.pYCoordList + WaldosWorld.r/2 * (-wL+wR)*np.sin(self.pThetaList)*WaldosWorld.deltaT
        self.pThetaList = self.pThetaList + WaldosWorld.r*(wL+wR)/WaldosWorld.l * WaldosWorld.deltaT
        #move back to Body Frame
        self.pXCoordList, self.pYCoordList = WaldosWorld.transform(self.pXCoordList, self.pYCoordList, self.pThetaList,
                                                       WaldosWorld.bodyCoords, WaldosWorld.axleCoords)
        #Process Noise
        self.pXCoordList = np.random.normal(self.pXCoordList, self.sigma[0])
        self.pYCoordList = np.random.normal(self.pYCoordList, self.sigma[1])
        self.pThetaList = np.random.normal(self.pThetaList, self.sigma[2])

        #Everything should be within the enclosure
        idx = self.pXCoordList < 30.0
        self.pXCoordList[idx] = 30.0
        idx = self.pXCoordList > 260.0
        self.pXCoordList[idx] = 260.0
        idx = self.pYCoordList < 30.0
        self.pYCoordList[idx] = 30.0
        idx = self.pYCoordList > 260.0
        self.pYCoordList[idx] = 260.0
        return

    def data_association(self):
        '''ParticleFilter.data_association()
        Finds which observations correspond to which landmarks'''
        pass

    def update_weights(self, d, az, lidar):  
        #First move the particles to LiDAR Frame
        if lidar == 'RP':
            self.pXCoordList, self.pYCoordList = WaldosWorld.transform(self.pXCoordList, self.pYCoordList, self.pThetaList,
                                                       WaldosWorld.rpCoords, WaldosWorld.bodyCoords)
        else: # TF Luna is installed at Axel Coords
            self.pXCoordList, self.pYCoordList = WaldosWorld.transform(self.pXCoordList, self.pYCoordList, self.pThetaList,
                                                       WaldosWorld.axleCoords, WaldosWorld.bodyCoords)
    
        x = len(d)
        '''
        if x>10:
            x= 10
        '''
        for i in range(x):
            self.pPhiList.fill(az[i])
            distList = np.array([-1.0]*self.nParticles)
            for landmark in wrt.get_lidar_landmarks():
                landmark.find_distance(self.pXCoordList, self.pYCoordList, self.pThetaList+self.pPhiList, distList)
            
            #print(self.pThetaList+self.pPhiList)
            #print(self.get_pData())
            #idx = distList == None
            #print(self.pXCoordList[idx], self.pYCoordList[idx])
            #print(d[i])
            prob = norm.pdf(d[i],distList, self.sigmaLandmark)
     #       print(distList, d[i], az[i], prob)
            self.weightList *= prob
            #print(np.max(self.weightList))
            if np.max(self.weightList) < 1e-100:
             #   print('All particle weight reset')
                self.weightList.fill(1)
        self.weightList = self.weightList/np.sum(self.weightList) # normalize
            
        #move the particles back to Body Frame
        if lidar == 'RP':
            self.pXCoordList, self.pYCoordList = WaldosWorld.transform(self.pXCoordList, self.pYCoordList, self.pThetaList,
                                                       WaldosWorld.bodyCoords, WaldosWorld.rpCoords)
        else: # TF Luna is installed at Axel Coords
            self.pXCoordList, self.pYCoordList = WaldosWorld.transform(self.pXCoordList, self.pYCoordList, self.pThetaList,
                                                       WaldosWorld.bodyCoords, WaldosWorld.axleCoords)
        

                
    def resample(self):
        '''ParticleFilter.resample()
        Resamples from the udpated set of particles to form the new set of particles'''
        '''ParticleFilter.resample()
        Resamples fromm the udpated set of particles to form the new set of particles'''
        #self.weightList = self.weightList/np.sum(self.weightList) #normalize weightList
        resampledIndices = np.random.choice(self.nParticles, self.nParticles, p = self.weightList)
        
        resampledPXCoordList = self.pXCoordList[resampledIndices]
        resampledPYCoordList = self.pYCoordList[resampledIndices]
        resampledthetaList = self.pThetaList[resampledIndices]
        resampledphiList = self.pPhiList[resampledIndices]
        resampledWeightList = self.weightList[resampledIndices]
        (uniqueIndices, counts) = np.unique(resampledIndices,return_counts = True)
        print(uniqueIndices.size)
        '''
        #if uniqueIndices.size < 50:
        #    return
        resampledPXCoordList = self.pXCoordList[uniqueIndices]
        resampledPYCoordList = self.pYCoordList[uniqueIndices]
        resampledthetaList = self.pThetaList[uniqueIndices]
        resampledphiList = self.pPhiList[uniqueIndices]
        resampledWeightList = self.weightList[uniqueIndices]*counts
        self.nParticles = uniqueIndices.size
        #print(resampledWeightList)
        '''
        self.pXCoordList = resampledPXCoordList
        self.pYCoordList = resampledPYCoordList
        self.pThetaList = resampledthetaList
        self.pPhiList = resampledphiList
        self.weightList = resampledWeightList/np.sum(resampledWeightList)
        return


# setup paramaters and logging
'''
start = 1676760244.3 - 1
end = 1676760297.9 + 1

fl = 'Data/HR_M_RP_W21_C1leftMotor.npy'
fr = 'Data/HR_M_RP_W21_C1rightMotor.npy'
frp = 'Data/HR_M_RP_W21_C1RP.npy'
'''

'''
ft = 'Data/Video/M_RP_W15_O270_C_2timer.npy'
fl = 'Data/Video/M_RP_W15_O270_C_2leftMotor.npy'
fr = 'Data/Video/M_RP_W15_O270_C_2rightMotor.npy'
frp = 'Data/Video/M_RP_W15_O270_C_2RP.npy'
'''

'''
ft = 'Data/Video/M_RP_W15_O270_C_2timer.npy'
fl = 'Data/Video/M_RP_W15_O270_C_2leftMotor.npy'
fr = 'Data/Video/M_RP_W15_O270_C_2rightMotor.npy'
frp = 'Data/Video/M_RP_W15_O270_C_2RP.npy'

t=np.load(ft, allow_pickle = True)
start = t[0]
end = t[1]
'''

fl = 'Data/Video/HR_M_RP_W21_S1leftMotor.npy'
fr = 'Data/Video/HR_M_RP_W21_S1rightMotor.npy'
frp = 'Data/Video/HR_M_RP_W21_S1RP.npy'
start = 1676761018.15559
end = 1676761097.3452275

#align everything to 100 msec boundary
deltaT = WaldosWorld.deltaT
abs_time = np.arange(np.round(start,1), np.round(end,1)+deltaT, deltaT)
abs_time = np.round(abs_time,1) # round for intersect1d

wrt = WaldoRaceTrack()
nParticles = 100
initState = np.array([*WaldosWorld.WP21, np.pi])
#initState = np.array([*WaldosWorld.WP15, 3.0*np.pi/2.0])
sigma = np.array([5.0, 5.0, 0.01]) #cm, cm, radians
pf = ParticleFilter(nParticles, wrt, initState, sigma)
#pf.set_pCoord(0, initState)

dataL = np.load(fl, allow_pickle = True)
dataR = np.load(fr, allow_pickle = True)
wL, wR, wL1, wR1 = GTWheelOdometry.get_motor_speeds(abs_time, dataL, dataR)

t, dist, azimuth = load_RPLidar_data(frp)

bState = []
wState = []
dataFrames = []
for index, abs_t in enumerate(abs_time):
    #if index != 11:
    #    continue
    print('processing '+str(index))
    pf.prediction(wL[index], wR[index])
    d, az = get_RPLidar_data(t, dist, azimuth, abs_t)
    if len(d) == 0:
        continue
    pf.update_weights(d, az, 'RP')
    pf.resample()
    print('processing done')
    pDF = pf.get_pData()
    dataFrames.append(pDF)
    max_weight = pDF['weight'].max()
    idx = np.where(pDF['weight']== max_weight)[0]
    pDFBest = pDF.loc[idx]
    bX = pDFBest['x'].iloc[0]
    bY = pDFBest['y'].iloc[0]
    btheta = pDFBest['theta'].iloc[0]
    bState.append([bX, bY, btheta])
    wX = np.sum(pDF['x']*pDF['weight'])/np.sum(pDF['weight'])
    wY = np.sum(pDF['y']*pDF['weight'])/np.sum(pDF['weight'])
    wtheta = np.sum(pDF['theta']*pDF['weight'])/np.sum(pDF['weight'])
    wState.append([wX, wY, wtheta])
   


bState = np.array(bState)
wState = np.array(wState)

ax = WaldosWorld.plot_RectView('Particle Filter Localization Results')
ax.plot(wState[:,0],wState[:,1], label = 'Weighted Particle', c = 'cyan')
ax.plot(bState[:,0],bState[:,1], label = 'Best Particle', c = 'crimson')
plt.legend()
plt.savefig('W21Localization2.jpg')
plt.show(block = False)


'''
#plt.rcParams["figure.figsize"] = [7.50, 3.50]
#plt.rcParams["figure.autolayout"] = True
def getImage(path):
   return OffsetImage(plt.imread(path, format="jpg"), zoom=1)

for i in range(len(bState)):
    ax1 = WaldosWorld.plot_RectView('Particle Filter Localization Results, Best Particle')
    ab = AnnotationBbox(getImage('WaldoMarker.jpg'), (bState[i,0],bState[i,1]), frameon=False)
    ax1.add_artist(ab)
    #ax1.scatter(wState[i,0],wState[i,1], marker = '.', c = 'green')
    #ax1.scatter(bState[i,0],bState[i,1], marker = 'x',c = 'crimson')
    ax1.plot(bState[0:i,0],bState[0:i,1],c = 'crimson')
    fstr = 'VideoMaterial/Positions/Pos{}.jpg'.format(i)
    plt.savefig(fstr)
    plt.close()
    print(i)
'''




'''
for i in range(len(bState)):
    ax2 = WaldosWorld.plot_RectView()
    ax2.scatter(dataFrames[i]['x'],dataFrames[i]['y'], marker = '.', c='orange')
    fstr = 'VideoMaterial/Particles/Par{}.jpg'.format(i)
    plt.savefig(fstr)
#plt.show(block=False)


fileStr = 'Data/test_w01RP.npy'

get_RPLidar_data = load_RPLidar_data(fileStr)
data = next(get_RPLidar_data)
st, et, d, az = data
ax = WaldosWorld.plot_LiDARView(146, 31, 90, 'RPLiDAR View')
ax.plot(az,d)
plt.show(block=False)

'''    
