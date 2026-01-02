#!/usr/bin/env python3
'''Measures sensor scanning speed'''
import TFLunaLidar

PORT_NAME = '/dev/ttyUSB0'
speed = 50
sampleRate = 100

def run():
    '''Main function'''
    data = []
    sampleData = []
    print('Running Scan Rate and Sample Rate Test for 100 scans') 
    print('First few scans')
    luna_iterscans = TFLunaLidar.tfluna_iterscans(PORT_NAME, sampleRate, speed)
    i = 0
    start_time, end_time, scan = next(luna_iterscans) # Throw away first scan
    txt = ''
    while True:
        start_time, end_time, scan = next(luna_iterscans)
        i+=1
        delta = end_time-start_time
        scanLen = len(scan)
        data.append(delta)
        sampleData.append(scanLen/delta)
        if i < 4:
            print(' Scan Time: %1.3f, Scan %d: %d Measurments' % (end_time-start_time, i, len(scan)))
        else:
            txt += '.'
        if i > 9:
            break
        
    TFLunaLidar.tfluna_stop('test')
    print(txt)
    print('Stoping. Computing mean...')
    print('TFLuna Lidar run at Motor Speed '+str(speed)+' Sample Rate ' + str(sampleRate))
    print('Scan Rate, TurnTable Speed, Sample Rate')
    delta = sum(data)/len(data)
    meanSample = sum(sampleData)/len(sampleData)
    print('Mean: %.2f Hz, %.2f RPM, %.2f Samples/Sec' % (1/delta, 60/delta, meanSample))

if __name__ == '__main__':
    run()
