import numpy as np
import matplotlib.pyplot as plt

# All distance units cm

sqrt2 = np.sqrt(2)

#Waldo's Racetrack Coordinates
coord0 = (0.0,0.0)
coord1 = (0.0,259.1)
coord2 = (61.0, 289.6)
coord3 = (259.1, 289.6)
coord4 = (289.6, 228.6)
coord5 = (289.6, 30.5)
coord6 = (259.1, 0.0)

#Wall Coordinates, Slopes and Coefficients of Lines
wallCoords = np.array([coord0,coord1,coord2,coord3,coord4,coord5,coord6,coord0])

infSlope = np.tan(np.pi/2)
wallBetaArr = np.array([infSlope, 0.5, 0, -2, infSlope, 1, 0])
wallCoeffArr = np.array([[infSlope, -1], [0.5, -1], [0, -1], [-2, -1], [infSlope, -1], [1, -1], [0, -1]])

#Circle Parameters
center = np.array([141.0, 141.0])
radius = 96.3
step = 0.001
#Circle and Square Coordinates
theta = np.arange(0,2*np.pi + step,step)
xCoord = center[0] + radius*np.cos(theta)
yCoord = center[1] + radius*np.sin(theta)

circle_coords = np.array([xCoord, yCoord])
square_coords = np.array([center + radius, np.array([center[0] + radius, center[1] - radius]),
                          center - radius, np.array([center[0] - radius, center[1] +radius]), center+radius])

#Waypoint Coordinates
a = 6.3
b = 45.0
c = 51.0
d = 44.7
WP1 = (c + b*2 + 5, d + a)
WP2 = (WP1[0], WP1[1] + b)
WP3 = (WP1[0], WP1[1] + b*2)
WP4 = (WP1[0], WP1[1] + b*3)
WP5 = (WP1[0], WP1[1] + b*4)
WP16 = (c + b*4, d)
WP17 = (c + b*3, d)
WP18 = (c + b*2, d)
WP19 = (c + b, d)
WP20 = (c, d)
WP6 = (WP16[0] + a, WP16[1] + a)
WP7 = (WP16[0] + a, WP16[1] + a + b*1)
WP8 = (WP16[0] + a, WP16[1] + a + b*2)
WP9 = (WP16[0] + a, WP16[1] + a + b*3)
WP10 = (WP16[0] + a, WP16[1] + a + b*4)
WP11 = (WP20[0] - a, WP20[1] + a)
WP12 = (WP20[0] - a, WP20[1] + a + b*1)
WP13 = (WP20[0] - a, WP20[1] + a + b*2)
WP14 = (WP20[0] - a, WP20[1] + a + b*3)
WP15 = (WP20[0] - a, WP20[1] + a + b*4)
WP21 = (WP10[0] - a, WP10[1] + a)
WP22 = (WP10[0] - a - b*1, WP10[1] + a)
WP23 = (WP10[0] - a - b*2, WP10[1] + a)
WP24 = (WP10[0] - a - b*3, WP10[1] + a)
WP25 = (WP10[0] - a - b*4, WP10[1] + a)
WP26 = (center[0] + radius/sqrt2, center[1] - radius/sqrt2)
WP27 = (center[0] - radius/sqrt2, center[1] - radius/sqrt2)
WP28 = (center[0] + radius/sqrt2, center[1] + radius/sqrt2)
WP29 = (center[0] - radius/sqrt2, center[1] + radius/sqrt2)

wpCoords = np.array([WP1, WP2, WP3, WP4, WP5, WP6, WP7, WP8, WP9, WP10,
                     WP11, WP12, WP13, WP14, WP15, WP16, WP17, WP18, WP19, WP20,
                     WP21, WP22, WP23, WP24, WP25, WP26, WP27, WP28, WP29])

# Frames in Waldo Body Frame
# (0,0) is at the front of Waldo, marked by Yellow Triangles
bodyCoords = np.array([0.0, 0.0])
cameraCoords = np.array([-0.8, 7.6]) 
rpCoords = np.array([-20.0, 0.0])
colorCoords = np.array([-14.4, 0.0])
axleCoords = np.array([-18.4, 0.0])
lunaCoords = np.array([-18.4, 0.0])

refBCoords = np.array([WP2[0]-5, WP2[1]-16])
refACoords = np.array([WP4[0]-5, WP4[1]+16])

r = 4.4 # wheel radius in cm
l = 14.2 # axle length in cm
max_motor_speed = 175*2*np.pi/60 # rad/sec

deltaT = 0.1 #10Hz rate

#ax = plot_results(wallCoords,wpCoords, circle_coords, square_coords)

def transform(x, y, theta, toFrame, fromFrame):
    '''
    Transform a np.array of points from  one frame to another
        assumes theta in radians
    '''
    t = toFrame - fromFrame
    cosTheta = np.cos(theta)
    sinTheta = np.sin(theta)
    xm = x + t[0]*cosTheta - t[1]*sinTheta
    ym = y + t[0]*sinTheta + t[1]*cosTheta
    return xm, ym

def plot_RectView(titleString = 'Waldo\'s World'):
    fig, ax = plt.subplots()
    ax.set_title(titleString, fontsize = 10)
    ax.set_xlabel('cm')
    ax.set_ylabel('cm')
    ax.plot(wallCoords[:,0],wallCoords[0:,1])
    ax.scatter(wpCoords[:,0], wpCoords[:,1],marker = '+', c='black')
    ax.plot(circle_coords[0], circle_coords[1], c = 'black')
    ax.plot(square_coords[:,0], square_coords[:,1], c = 'black')
    ax.set_aspect('equal')
    ax.set_xlim(right = 300)
    ax.set_ylim(top = 300)
    #plt.show(block = False)
    plt.grid()
    return ax

def find_distance(x, y, theta):
    #Theta is in radians, function needs it in degrees
    theta = (theta*180/np.pi) % 360
    xCoords = wallCoords[:,0]-x
    yCoords = wallCoords[:,1]-y
    angles = ((np.arctan2(yCoords, xCoords) * 180 / np.pi)+360)%360
    for i in range(len(wallBetaArr)):
        if np.abs(angles[i]-angles[i+1]) > 180:
            big = max(angles[i],angles[i+1])
            small = min(angles[i],angles[i+1])
            if 0 <= theta <= small or big <= theta <= 360:
                coeff = np.array([np.tan(theta/180*np.pi), -1])
                a = np.stack((wallCoeffArr[i], coeff), axis = 0)
                b = np.array([-yCoords[i] + wallBetaArr[i]*xCoords[i], 0])
                z = np.linalg.solve(a, b)
                dist = np.sqrt(np.dot(z,z))
                return dist
        elif angles[i]<= theta <= angles[i+1] or  angles[i+1]<= theta <= angles[i]:
            coeff = np.array([np.tan(theta/180*np.pi), -1])
            a = np.stack((wallCoeffArr[i], coeff), axis = 0)
            b = np.array([-yCoords[i] + wallBetaArr[i]*xCoords[i], 0])
            z = np.linalg.solve(a, b)
            dist = np.sqrt(np.dot(z,z))
            return dist


def plot_LiDARView(x, y, theta, titleStr = 'LiDAR View'):
    ''' theta in degrees'''
    az = np.arange(360)
    distances = []
    for angle in az:
        distances.append(find_distance(x, y, angle*np.pi/180))
    az = ((np.arange(360)-theta)+360)%360
    fig, ax = plt.subplots(subplot_kw = {'projection': 'polar'})
    ax.plot(az*np.pi/180, distances)
    ax.set_title(titleStr, fontsize = 10)
    plt.show(block = False)
    return ax


def plot_DistanceView(titleString = 'Waldo\'s World'):
    fig, ax = plt.subplots()
    ax.set_title(titleString, fontsize = 10)
    ax.set_xlabel('cm')
    ax.set_ylabel('cm')
    ax.plot(wallCoords[:,0]-WP3[0],wallCoords[0:,1]-WP3[1])
    ax.plot([-30,30],[0,0], c='orange')
    ax.plot([0,0],[-30,30], c='orange')
    ax.plot([0,wallCoords[0,0]-WP3[0]],[0, wallCoords[0,1]-WP3[1]], c = 'green')
    ax.scatter([0,wallCoords[0,0]-WP3[0]],[0, wallCoords[0,1]-WP3[1]], c = 'green', marker = '.')
    ax.plot([0,wallCoords[1,0]-WP3[0]],[0, wallCoords[1,1]-WP3[1]], c = 'green')
    ax.plot([0,wallCoords[2,0]-WP3[0]],[0, wallCoords[2,1]-WP3[1]], c = 'green')
    ax.plot([0,wallCoords[3,0]-WP3[0]],[0, wallCoords[3,1]-WP3[1]], c = 'green')
    ax.plot([0,wallCoords[4,0]-WP3[0]],[0, wallCoords[4,1]-WP3[1]], c = 'green')
    ax.plot([0,wallCoords[5,0]-WP3[0]],[0, wallCoords[5,1]-WP3[1]], c = 'green')
    ax.plot([0,wallCoords[6,0]-WP3[0]],[0, wallCoords[6,1]-WP3[1]], c = 'green')
    ax.scatter([0,wallCoords[1,0]-WP3[0]],[0, wallCoords[1,1]-WP3[1]], c = 'green', marker = '.')
    ax.scatter([0,wallCoords[2,0]-WP3[0]],[0, wallCoords[2,1]-WP3[1]], c = 'green', marker = '.')
    ax.scatter([0,wallCoords[3,0]-WP3[0]],[0, wallCoords[3,1]-WP3[1]], c = 'green', marker = '.')
    ax.scatter([0,wallCoords[4,0]-WP3[0]],[0, wallCoords[4,1]-WP3[1]], c = 'green', marker = '.')
    ax.scatter([0,wallCoords[5,0]-WP3[0]],[0, wallCoords[5,1]-WP3[1]], c = 'green', marker = '.')
    ax.scatter([0,wallCoords[6,0]-WP3[0]],[0, wallCoords[6,1]-WP3[1]], c = 'green', marker = '.')
    ax.plot([0,50],[0,wallCoords[2,1]-WP3[1]], c='crimson')
    ax.scatter([0,50],[0,wallCoords[2,1]-WP3[1]], c='crimson', marker = '.')
    angle = np.arctan((wallCoords[2,1]-WP3[1])/50.0)
    x = []
    y = []
    for a in np.arange(0,angle + 0.01,0.01):
        x.append(25*np.cos(a))
        y.append(25*np.sin(a))
    ax.plot(x,y,c='crimson')
    angle = np.pi*2+np.arctan2(wallCoords[0,1]-WP3[1],wallCoords[0,0]-WP3[0])
    x = []
    y = []
    for a in np.arange(0,angle + 0.01,0.01):
        x.append(15*np.cos(a))
        y.append(15*np.sin(a))
    ax.plot(x,y,c='green')
    
    ax.annotate(r'$\theta + \phi$',xy = (25,25), rotation = 45,c = 'crimson')
    #ax.scatter(wpCoords[:,0], wpCoords[:,1])
    #ax.plot(circle_coords[0], circle_coords[1])
    #ax.plot(square_coords[:,0], square_coords[:,1])
    ax.annotate('Sector 0',xy = (-125,0), c = 'green')
    ax.annotate('Sector 1',xy = (-100,80), rotation = -60, c = 'green')
    ax.annotate('Sector 2',xy = (-25,100), c = 'green')
    ax.annotate('Sector 3',xy = (80,80), rotation = 45, c = 'green')
    ax.annotate('Sector 4',xy = (80,0), c = 'green')
    ax.annotate('Sector 5',xy = (80,-120), rotation = -45, c = 'green')
    ax.annotate('Sector 6',xy = (-25,-100), c = 'green')
    ax.annotate(r'$l_{0}$',xy = (-158,-5), rotation = 90, c = '#1f77b4')
    ax.annotate(r'$l_{1}$',xy = (-125,140), rotation = 26, c = '#1f77b4')
    ax.annotate(r'$l_{2}$',xy = (0,152), c = '#1f77b4')
    ax.annotate(r'$l_{3}$',xy = (125,120), rotation = -64, c = '#1f77b4')
    ax.annotate(r'$l_{4}$',xy = (145,0), rotation = -90, c = '#1f77b4')
    ax.annotate(r'$l_{5}$',xy = (130,-130), rotation = -90, c = '#1f77b4')
    ax.annotate(r'$l_{6}$',xy = (-25,-153), c = '#1f77b4')
    ax.annotate(r'$l_{m}$',xy = (25,75), rotation = angle*180/np.pi, c = 'crimson')
    ax.set_aspect('equal')
    ax.annotate(r'$(x_{0},y_{0})$',xy = (-164,-150), c = 'green')
    ax.annotate(r'$\alpha_{0} = arctan2(y_{0},x_{0})$',xy = (-125,-100), rotation = 45, c = 'green')
    ax.annotate(r'$\alpha_{1}$',xy = (-155,125), c = 'green')
    ax.annotate(r'$\alpha_{2}$',xy = (-85,152), c = 'green')
    ax.annotate(r'$\alpha_{3}$',xy = (116,150), c = 'green')
    ax.annotate(r'$\alpha_{4}$',xy = (145,80), c = 'green')
    ax.annotate(r'$\alpha_{5}$',xy = (145,-115), c = 'green')
    ax.annotate(r'$\alpha_{6}$',xy = (115,-148), c = 'green')
    ax.set_xlim(-165, 165)
    ax.set_ylim(-165, 165)
    plt.show(block = False)
    #plt.grid()
    #plt.savefig('FindDistance.jpg')
    return ax

#ax = plot_DistanceView()
    
