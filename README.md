# Localizing-Waldo

A particle filter implementation for robot localization using Bayesian inference. This repository contains code from two science fair projects exploring how a LEGO-based robot can determine its position within an environment using various sensors.

---

## Repository Structure

```
Localizing-Waldo/
├── Omnidirectional Lidar Data Acquisition/    # Raspberry Pi data collection scripts
├── Localizing Waldo 1 (Color & Ultrasonic Sensors)/  # Particle filter with LEGO sensors
├── Localizing Waldo 2 (LiDAR & Camera)/       # Particle filter with LiDAR and camera
├── Test Bed Image Creation/                    # Scripts to generate test bed floor images
├── requirements.txt
└── README.md
```

---

## Omnidirectional Lidar Data Acquisition

Scripts for synchronized sensor data collection on Raspberry Pi with BlueDot remote control.

| File | Description |
|------|-------------|
| `MovingRPLidarAcquisition.py` | Acquires RPLidar scans, motor encoder data, color sensor data, and video while robot moves. BlueDot app controls the robot over Bluetooth. |
| `MovingTFLunaLidarAcquisition.py` | Same as above but for TFLuna LiDAR. Includes IR beam breaker interrupt handling for tracking turntable rotations. |
| `StationaryRPLidarAcquisition.py` | Records 5 RPLidar scans and captures a photo at a stationary position. |
| `StationaryTFLunaLidarAcquisition.py` | Records 5 TFLuna scans and captures a photo at a stationary position. |
| `RPLidarParams.py` | Measures RPLidar scan rate (Hz), motor speed (RPM), and sample rate (samples/sec). |
| `TFLunaParams.py` | Measures TFLuna scan rate and sample rate for the turntable-based 3D LiDAR setup. |
| `RPLidarDemo.py` | Real-time visualization of RPLidar data using pygame on a 320x240 display. |
| `TFLunaLidarDemo.py` | Real-time visualization of TFLuna data with turntable motor control and break-beam sensor for revolution tracking. |
| `cameratest.py` | Utility for capturing camera calibration images. |

---

## Localizing Waldo 1 (Color and Ultrasonic Sensors)

Particle filter localization using LEGO Color Sensor and Ultrasonic Sensor on a 1200mm x 1200mm "Color World" test bed with 900 randomly colored squares.

| File | Description |
|------|-------------|
| `ParticleFilterImproved.py` | Vectorized particle filter using NumPy. Supports up to 1 million particles. Implements prediction (motion model with Gaussian noise), weight update (color and ultrasonic likelihood), and resampling steps. |
| `ParticleFilter.py` | Original list-based particle filter implementation. Includes functions for evaluating color sensor and ultrasonic sensor performance. |
| `LocalizationPerformance.py` | Generates CDF plots and average time statistics for localization results across different particle counts and sensor configurations. |
| `main.py` | EV3 Brick motor control script using pybricks-micropython. |
| `Map.py` | Placeholder for map representation. |
| `InferenceEngine.py` | Placeholder for histogram filter based on color sensor. |
| `VideoPlayer.py` | Utility for video playback in notebooks. |

### Key Classes (ParticleFilterImproved.py)
- `WaldoWorld`: Defines the test enclosure geometry (8 walls) and color board coordinates
- `ColorWorld`: Generates 30x30 random color grid with test site locations
- `USLandmark`: Computes distances from particles to walls for ultrasonic measurements
- `ParticleFilter`: Implements prediction, weight update (CS/US), and resampling

---

## Localizing Waldo 2 (LiDAR and Camera)

Particle filter localization using RPLidar A1 or TFLuna LiDAR with a camera-based ground truth method. Test bed is a 290cm x 290cm "Waldo's Racetrack" with circular and rectangular tracks.

| File | Description |
|------|-------------|
| `PF.py` | Vectorized particle filter with kinematic motion model. Synchronizes measurements to 100ms ticks. Implements LiDAR measurement likelihood using wall distance computations. |
| `WaldosWorld.py` | Defines racetrack geometry, 29 waypoints, robot frame transformations (body, camera, axle, LiDAR frames), and visualization functions. |
| `GTCamera.py` | Camera-based ground truth localization. Undistorts images, finds circle features using Hough transform, clusters circles into L-shaped landmarks, uses CNN to identify landmark IDs, computes heading and position relative to reference frames. |
| `LandmarkClassifier.py` | Keras CNN (LeNet-based) for landmark digit recognition. 64x64 grayscale input, 5 classes, trained on 5000+ images with 99.6% test accuracy. |
| `GTWheelOdometry.py` | Implements robot kinematic motion model from motor encoder data. Computes angular velocities and integrates to get position trajectory. |
| `RPLidar.py` | Computes RPLidar measurement errors compared to expected distances at known positions. |
| `TFLuna.py` | Computes TFLuna measurement errors compared to expected distances at known positions. |
| `CalibrateCamera.py` | Camera calibration using chessboard pattern to obtain camera matrix and distortion coefficients. |

### Key Classes (PF.py)
- `WaldoRaceTrack`: Defines 7 walls with slope coefficients for distance computations
- `LiDARLandmark`: Computes expected LiDAR distances for each particle state
- `ParticleFilter`: Motion prediction using wheel odometry model, weight update using LiDAR likelihood, resampling

### Data Format
All sensor data saved as NumPy `.npy` files:
- Timer: `[start_time, end_time]`
- Motor: `[[timestamp, speed%, relative_angle, absolute_angle], ...]`
- RPLidar: `[[scan_id, start_time, end_time, [[quality, angle, distance], ...]], ...]`
- TFLuna: `[[scan_id, timestamp, distance, strength, temperature], ...]`

---

## Test Bed Image Creation

Scripts to generate printable floor images for the test beds.

| File | Description |
|------|-------------|
| `Waldo1_ColorWorld_ImageCreation.ipynb` | Generates 30x30 color grid (40mm squares) with test site markers. Outputs 36 pages at 200dpi for printing on vinyl stickers. |
| `Waldo2_RaceTrack_ImageCreation.py` | Generates 221cm x 221cm floor image with circular/rectangular tracks, 30 numbered waypoints, and track outlines. Outputs segmented blocks for printing. |

---

## Dependencies

**Core:**
- numpy, pandas, matplotlib, scipy, opencv-python

**Deep Learning (Waldo 2):**
- tensorflow, keras, scikit-learn

**Raspberry Pi Hardware:**
- rplidar, picamera, buildhat, bluedot, RPi.GPIO, tfluna

**Visualization:**
- pygame

---

## Hardware

**Robot Platform:**
- Raspberry Pi 3B+/4 with Build HAT
- LEGO Technic motors and wheels
- Waldo 1: LEGO Color Sensor, LEGO Ultrasonic Sensor
- Waldo 2: Slamtec RPLidar A1 or TFLuna on 2-axis turntable, Raspberry Pi Camera

**Test Beds:**
- Waldo 1: 1800mm x 1800mm enclosure with 1200mm x 1200mm color grid
- Waldo 2: 290cm x 290cm enclosure with 30cm walls, 5 ceiling landmarks for camera ground truth
