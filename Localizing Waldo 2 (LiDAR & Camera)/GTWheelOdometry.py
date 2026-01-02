import numpy as np
import matplotlib.pyplot as plt 
import WaldosWorld

deltaT = WaldosWorld.deltaT

def get_motor_speeds(abs_time, dataL, dataR):
    ltime = np.round(dataL[:,0],1)
    cl,idxl1,idxl2 = np.intersect1d(abs_time,ltime,return_indices = True)
    rtime = np.round(dataR[:,0],1)
    cr,idxr1,idxr2 = np.intersect1d(abs_time,rtime,return_indices = True)
    nSamples = len(abs_time)
    thetaL = np.array([0.0]*nSamples)
    thetaR = np.array([0.0]*nSamples)

    idx = idxl1[0]
    thetaL[:idx] = dataL[0][2]
    idx = idxl1[-1]
    thetaL[idx:] = dataL[-1][2]

    for i in range(len(idxl1)-1):
        thetaL[idxl1[i]:idxl1[i+1]] = dataL[idxl2[i]][2]
    
    idx = idxr1[0]
    thetaR[:idx] = dataR[0][2]
    idx = idxr1[-1]
    thetaR[idx:] = dataR[-1][2]

    for i in range(len(idxr1)-1):
        thetaR[idxr1[i]:idxr1[i+1]] = dataR[idxr2[i]][2]
    
    wL = np.array([0.0]*nSamples)
    wR = np.array([0.0]*nSamples)
    wL1 = np.array([0.0]*nSamples)
    wR1 = np.array([0.0]*nSamples)
    # Get Angles
    thetaL[idxl1] = dataL[:,2][idxl2]
    thetaR[idxr1] = dataR[:,2][idxr2]
    # Get Omega
    wL[1:nSamples] = (thetaL[1:nSamples] - thetaL[0:nSamples - 1])*np.pi/(180*deltaT)
    wR[1:nSamples] = (thetaR[1:nSamples] - thetaR[0:nSamples - 1])*np.pi/(180*deltaT)
    wL1[idxl1] = dataL[:,1][idxl2]*WaldosWorld.max_motor_speed/100
    wR1[idxr1] = dataR[:,1][idxr2]*WaldosWorld.max_motor_speed/100
    return wL, wR, wL1, wR1

def wheelOdomtetry(fl, fr, start, end, initState):
    dataL = np.load(fl, allow_pickle = True)
    dataR = np.load(fr, allow_pickle = True)
    abs_time = np.arange(np.round(start,1), np.round(end,1)+deltaT, deltaT)
    abs_time = np.round(abs_time,1) # round for intersect1d
    wL, wR, wL1, wR1 = get_motor_speeds(abs_time, dataL, dataR)

    nSamples = len(abs_time)
    states = np.array([None, None, None]*nSamples).reshape(-1,3)
    states[0,0:2] = WaldosWorld.transform(initState[0], initState[1], initState[2],WaldosWorld.axleCoords, WaldosWorld.bodyCoords)
    states[0,2] = initState[2]
    # state = (x,y,theta)
    for index, abs_t in enumerate(abs_time):
        if index == 0:
            continue
        states[index,0] = states[index-1,0] + WaldosWorld.r/2*(-wL[index-1]+wR[index-1])*np.cos(states[index-1,2])*deltaT
        states[index,1] = states[index-1,1] + WaldosWorld.r/2*(-wL[index-1]+wR[index-1])*np.sin(states[index-1,2])*deltaT
        states[index,2] = states[index-1,2] + WaldosWorld.r*(wL[index-1]+wR[index-1])/WaldosWorld.l * deltaT
    states = np.float32(states)
    states[:,0], states[:,1] = WaldosWorld.transform(states[:,0], states[:,1], states[:,2],WaldosWorld.bodyCoords, WaldosWorld.axleCoords) 
    return states

'''
ft = 'Data/Video/M_RP_W15_O270_C_2timer.npy'
fl = 'Data/Video/M_RP_W15_O270_C_2leftMotor.npy'
fr = 'Data/Video/M_RP_W15_O270_C_2rightMotor.npy'
frp = 'Data/Video/M_RP_W15_O270_C_2RP.npy'

t=np.load(ft, allow_pickle = True)
start = t[0]
end = t[1]

initState = np.array([*WaldosWorld.WP15, np.pi*270/180])
states = wheelOdomtetry(fl, fr, start, end, initState)
ax = WaldosWorld.plot_RectView('Wheel Odometry Localization')
ax.plot(states[:,0],states[:,1])
plt.show(block=False)
plt.savefig('WheelOdometry.jpg')  
'''


'''
#fl = 'Data/test_w21leftMotor.npy'
#fr = 'Data/test_w21rightMotor.npy'
'''



'''
r = WaldosWorld.r
l = WaldosWorld.l

def speed_aligned_100msec(data, abs_t):
    idx = data[:,0] >= abs_t
    m = np.where(idx==True)[0][0]
    return np.where(idx==True)[0][0]
    

max_motor_speed = 175*2*np.pi/60 # rad/sec
deltaT = 0.1

startL = dataL[0][0]
startR = dataR[0][0]
start = np.floor(np.maximum(startL, startR)*10)/10
endL = dataL[len(dataL)-1][0]
endR = dataR[len(dataR)-1][0]
end = np.floor(np.minimum(endL, endR)*10)/10

abs_time = np.arange(start, end+deltaT, deltaT)
#abs_time = abs_time[0:215]
nSamples = len(abs_time)

wL = np.array([None]*nSamples)
wR = np.array([None]*nSamples)
wL1 = np.array([None]*nSamples)
wR1 = np.array([None]*nSamples)
x = np.array([None]*nSamples)
y = np.array([None]*nSamples)
theta = np.array([None]*nSamples)

x_init = 249.0
y_init = 237.3
theta_init = np.pi

prev_i = -1
prev_j = -1
for index, abs_t in enumerate(abs_time):
    i = speed_aligned_100msec(dataL, abs_t)
    j = speed_aligned_100msec(dataR, abs_t)
    print(i,j)
    if index == 0:
        x[index] = x_init
        y[index] = y_init
        theta[index] = theta_init
        wL[index] = 0
        wR[index] = 0
    else:
        dtL = dataL[i,0]-dataL[i-1,0]
        if dtL > 0.125:
            wL[index] = 0
        else:
            wL[index] = (dataL[i,2]-dataL[i-1,2])*np.pi/(180*dtL)
        dtR = dataR[j,0]-dataR[j-1,0]
        if dtR > 0.125:
            print('here')
            wR[index] = 0
        else:
            wR[index] = (dataR[j,2]-dataR[j-1,2])*np.pi/(180*dtR)
        x[index] = x[index-1] + r/2*(-wL[index-1]+wR[index-1])*np.cos(theta[index-1])*deltaT
        y[index] = y[index-1] + r/2*(-wL[index-1]+wR[index-1])*np.sin(theta[index-1])*deltaT
        theta[index] = theta[index-1] + r*(wL[index-1]+wR[index-1])/l * deltaT
    prev_i = i
    prev_j = j

'''
