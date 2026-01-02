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
camera = PiCamera()
camera.resolution = (1640, 1232)

def run(fname):
    '''Main function'''
    lidar = RPLidar(PORT_NAME)
    #print(lidar.motor_speed)
    info = lidar.get_info()
    print('info')
    print(info)
    health = lidar.get_health()
    print('health')
    print(health)
    old_t = None
    data = []
    print('Recording RP Lidar Data')

    for i, scan in enumerate(lidar.iter_scans()):
        now = time.time()
        if old_t is None: # first scan, could be short, discard
            old_t = now
            continue
        data.append(np.array([i, old_t, now, np.array(scan)]))
        if i >= 5:
            break
                    
    print('RP Lidar Data Recording')
    lidar.stop()
    lidar.disconnect()
    np.save(fname+'.npy', np.array(data))
    camera.capture(fname +'.jpg')


if __name__ == '__main__':
    run('/home/pi/Data/Stationary/RP/'+sys.argv[1])
