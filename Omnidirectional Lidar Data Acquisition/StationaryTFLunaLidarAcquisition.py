import TFLunaLidar
''' Stationary Test for RPLidar
Record RPLidar scan in a numpy file and take a pic
Usage example:
stationary_rp_scans.py Test1
records Test1.npy and Test1.jpg '''

import sys
import numpy as np
from rplidar import RPLidar
from picamera import PiCamera
import time

PORT_NAME = '/dev/ttyUSB0'
speed = 60
sampleRate = 100
camera = PiCamera()
camera.resolution = (1640, 1232)
data = []
def run(fname):
    '''Main function'''
    luna_iterscans = TFLunaLidar.tfluna_iterscans(PORT_NAME, sampleRate, speed)
    print('Recording TFLuna Lidar Data')
    scan = next(luna_iterscans) # discard 1st scan
    for i in range(5):
        scan = next(luna_iterscans)
        data.append(scan)
    print('TFLuna Lidar Data Recorded')
    TFLunaLidar.tfluna_stop(fname+'TTMotor'+'.npy')
    np.save(fname+'Luna'+'.npy', np.array(data))
    camera.capture(fname +'.jpg')


if __name__ == '__main__':
    f = input('Filename: ')
    run('/home/pi/Data/Stationary/TfLuna/'+f)
