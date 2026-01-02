import os
from math import cos, sin, pi, floor
import pygame

import tfluna
import timeout_decorator

from buildhat import Motor
import time
import RPi.GPIO as GPIO
import threading

BEAM_PIN = 27

nRev = -1
time_t = time.time()
lock = threading.Lock()

tt_motor = Motor('A')


# Set up pygame and the display
os.putenv('SDL_FBDEV', '/dev/fb1')
pygame.init()
lcd = pygame.display.set_mode((320,240))
pygame.mouse.set_visible(False)
lcd.fill((0,0,0))
pygame.display.update()

# used to scale data to fit on the screen
max_distance = 0

#pylint: disable=redefined-outer-name,global-statement
def process_data(data):
    global max_distance
    lcd.fill((0,0,0))
    for angle in range(360):
        distance = data[angle]
        if distance > 0:                  # ignore initially ungathered data points
            max_distance = max([min([5000, distance]), max_distance])
            radians = angle * pi / 180.0
            x = distance * cos(radians)
            y = distance * sin(radians)
            point = (160 + int(x / max_distance * 119), 120 + int(y / max_distance * 119))
            lcd.set_at(point, pygame.Color(255, 255, 255))
    pygame.display.update()

def break_beam_callback(channel):
    global nRev
    global time_t
    global lock
    #print('r',revs)
    with lock:
        now = time.time()
        if now - time_t < 0.3:
            return
        nRev += 1
        time_t = time.time()
        print(nRev, time_t)

GPIO.setmode(GPIO.BCM)
GPIO.setup(BEAM_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.add_event_detect(BEAM_PIN, GPIO.RISING, callback=break_beam_callback)

def set_tfluna_atZero(speed):
    tt_motor.start(speed)
    while True:
        if GPIO.input(BEAM_PIN) == 0:
            tt_motor.stop()
            break
    tt_motor.run_to_position(0, speed, True)
    print(tt_motor.get())

def tfluna_iterscans():
    global nRev
    global time_t
    global lock
    with tfluna.TfLuna(baud_speed=115200, serial_name="/dev/ttyUSB0") as luna:
        luna.get_version()
        luna.set_samp_rate(250)
        
        set_tfluna_atZero(20)
        with lock:
            nRev = -1
        tt_motor.start(50)
        while True:
            with lock:
                startFlag = nRev != -1
            if startFlag:
                break
            continue
            
        while True:
            with lock:
                currScan = nRev
                start_time = time_t
            scanData = []
            while True:
                with lock:
                    scanFlag = currScan != nRev
                if scanFlag:
                    break
                now = time.time()
                distance,strength,temperature = luna.read_tfluna_data()
                scanData.append([currScan, now, distance, strength, temperature])
            with lock:
                end_time = time_t
            yield (start_time, end_time, scanData)
            
luna_iterscans = tfluna_iterscans()

scan_data = [0]*360

try:
    while True:
        start_time, end_time, scan = next(luna_iterscans)
        print(len(scan), end_time-start_time)
        for meas in scan:
            n, meas_time, distance, strength, temperature = meas
            if strength > 1000:
                angle = 360*(meas_time-start_time)/(end_time-start_time)
                scan_data[min([359, floor(angle)])] = distance
        process_data(scan_data)

except KeyboardInterrupt:
    print('Stoping.')
    tt_motor.stop()
