#!/usr/bin/env python3
'''Measures sensor scanning speed'''
from rplidar import RPLidar
import time

PORT_NAME = '/dev/ttyUSB0'

def run():
    '''Main function'''
    lidar = RPLidar(PORT_NAME)
    print('motor speed')
    print(lidar.motor_speed)
    info = lidar.get_info()
    
    print('info')
    print(info) 

    health = lidar.get_health()
    print('health')
    print(health)

    print('some scans')
    iter_scans = lidar.iter_scans()
    for i, scan in enumerate(iter_scans):
        print('Time: %6.3f, Scan %d: %d Measurments' % (time.time(), i, len(scan)))
        if i > 4:
            break

    old_t = None
    data = []
    sampleData = []
    try:
        print('Running Scan Rate and Sample Rate Test') 
        print('Press Ctrl+C to stop')
        for scan in iter_scans:
            now = time.time()
            if old_t is None:
                old_t = now
                continue
            delta = now - old_t
            scanLen = len(scan)
            #print('%.2f Hz, %.2f RPM' % (1/delta, 60/delta))
            #print('%.2f Hz, %.2f RPM, %.2f Sample/Sec' % (1/delta, 60/delta, scanLen/delta))
            data.append(delta)
            sampleData.append(scanLen/delta)
            old_t = now
    except KeyboardInterrupt:
        print('Stoping. Computing mean...')
        print('RPLidar A1')
        print('Scan Rate, Motor Speed, Sample Rate')
        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()
        delta = sum(data)/len(data)
        meanSample = sum(sampleData)/len(sampleData)
        print('Mean: %.2f Hz, %.2f RPM, %.2f Samples/Sec' % (1/delta, 60/delta, meanSample))

if __name__ == '__main__':
    run()