''' Moving Test for RPLidar
Records RPLidar scan in a numpy file and take a pic
Usage example:
MovingRPLidarAcquisition.py Test1
records Test1.npy, Test1Color.npy and Test1.h264 '''

import sys
import math
import numpy as np
from rplidar import RPLidar
from picamera import PiCamera
import time
import threading
from buildhat import Motor, ColorSensor
from bluedot import BlueDot
from signal import pause

PORT_NAME = '/dev/ttyUSB0'

camera = PiCamera()
camera.resolution = (1640, 1232)

color_sensor = ColorSensor('B')
motor_left = Motor('C')
motor_right = Motor('D')

dot = BlueDot()
timerArr = np.array([None, None])
maxSpeed = 40
testUnderway = False
fname = None
th = None
th2 = None

dpCount = 0 # Counts bluedot double presses

motor_left_data = []
motor_right_data = []

def handle_motor_left(speed, pos, apos):
    """Motor data
    :param speed: Speed of motor
    :param pos: Position of motor
    :param apos: Absolute position of motor
    """
    global motor_left_data
    motor_left_data.append([time.time(), speed, pos, apos])
    
def handle_motor_right(speed, pos, apos):
    """Motor data
    :param speed: Speed of motor
    :param pos: Position of motor
    :param apos: Absolute position of motor
    """
    global motor_right_data
    motor_right_data.append([time.time(), speed, pos, apos])

motor_right.when_rotated = handle_motor_right
motor_left.when_rotated = handle_motor_left

def get_sensor_data():
    global testUnderway
    data = []
    while testUnderway:
        now = time.time()
        color = color_sensor.wait_until_color("cyan")
        now = time.time()
        data.append(np.array([now, color]))
        #print(color + 'waypoint!')
    np.save(fname+'Color.npy', np.array(data))

def get_rplidar_data():
    global fname
    global testUnderWay
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
        if not testUnderway:
            break
                    
    print('RP Lidar Data Recorded')
    lidar.stop()
    lidar.disconnect()
    np.save(fname+'RP.npy', np.array(data))
#    lidar.stop_motor()

def start_test():
    global testUnderway
    global fname
    global th
    global th2
    print('starting test')
    testUnderway = True
    th = threading.Thread(target=get_rplidar_data)
    th.daemon = True
    th2 = threading.Thread(target=get_sensor_data)
    th2.daemon = True
    camera.start_recording(fname+'.h264')
    startTime = time.time()
    timerArr[0] = startTime
    print(startTime)
    th.start()
    th2.start()

def end_test():
    global testUnderway
    global th
    print('ending test')
    testUnderway = False
    th.join()
    th2.join()
    camera.stop_recording()
    stopTime = time.time()
    np.save(fname+'leftMotor.npy', np.array(motor_left_data))
    np.save(fname+'rightMotor.npy', np.array(motor_right_data))
    timerArr[1] = stopTime
    np.save(fname+'timer.npy',timerArr)
    print(stopTime)
    print('all done')
    
def stop_robot():
    motor_left.stop()
    motor_right.stop()

def move_robot(pos):
    if pos.distance < 0.2:
        return
    speed = maxSpeed * pos.distance
    angle = math.atan2(pos.y, pos.x)
    kp = 0.2
    if angle < 0: # go backwards
        speed = -1*speed
    correction = kp*(abs(angle) - math.pi/2)*maxSpeed
    leftspeed = speed-correction
    rightspeed = speed+correction
    #print(leftspeed,rightspeed)
    motor_left.start(-1 * leftspeed)
    motor_right.start(rightspeed)
    
def dp_func_handler():
    global dpCount
    
    if dpCount % 2 == 0:
        start_test()
    else:
        end_test()

    dpCount +=1
    
if __name__ == '__main__':
    f = input('Filename: ')
    fname = '/home/pi/Data/Moving/RP/'+f
    print('all set')
    dot.when_pressed = move_robot
    dot.when_released = stop_robot
    dot.when_moved = move_robot
    dot.when_double_pressed = dp_func_handler
    
    pause()
