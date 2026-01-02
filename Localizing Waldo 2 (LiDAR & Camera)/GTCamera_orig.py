import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob
import yaml
import itertools
import tensorflow as tf
import keras
import WaldosWorld
import time

pixels_per_cm =5.346455563123757
#5.3368
#5.33804997547807
#5.3453152054020325
#5.356075201955683
#5.354053420383207

pixels_per_cm_data = []

# Read camera matrix
with open("calibration_matrix.yaml", "r") as f:
    data = yaml.safe_load(f)
f.close()
mtx = np.array(data['camera_matrix'])
dist = np.array(data['dist_coeff'])

model = keras.models.load_model("LC_model.h5")

#img = cv.imread('./Calibration Images/Test39.jpg')
#h,  w = img.shape[:2]
(h, w) = (1232, 1640)
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

def find_circle_clusters(all_circs):
    clustered = []
    clusters = []
    clustersComplete = False
    if all_circs is None:
        return clusters
    start = -1
    newClusterIdx = -1
    newCluster = []
    for comb in itertools.combinations(range(len(all_circs[0])),2):
        #print('comb')
        #print(comb)
        if start != comb[0]:
            if len(newCluster) > 0:
                clusters.append(newCluster)
            newCluster = []
            start = comb[0]
        if newClusterIdx != comb[0] and comb[0] in clustered:
            continue
        elif newClusterIdx != comb[0] and comb[0] not in clustered: # start a new cluster
            newClusterIdx = comb[0]
            clustered.append(comb[0])
            newCluster.append(comb[0])
        if comb[1] not in clustered: # check if it is in this cluster
            x0,y0,r0 = all_circs[0][comb[0]]
            x1,y1,r1 = all_circs[0][comb[1]]
            #print(comb[0], comb[1])
            #print(x0,y0,x1,y1)
            if (x1-x0)**2 + (y1-y0)**2 < (20**2+30**2)* (pixels_per_cm+1)**2:
                #print('appending cluster')
                newCluster.append(comb[1])
                clustered.append(comb[1])
                #print(newCluster)
                #print(clustered)
    #print(clusters)
    #print(clustered)
    return clusters

def oriented_cluster(cluster, all_circs):
    for perm in itertools.permutations(cluster,3):
        x0, y0, r0 = all_circs[0][perm[0]]
        x1, y1, r1 = all_circs[0][perm[1]]
        x2, y2, r2 = all_circs[0][perm[2]]
        d0 = (x1-x0)**2 + (y1-y0)**2
        d1 = (x2-x1)**2 + (y2-y1)**2
        d2 = (x2-x0)**2 + (y2-y0)**2
        if d0 < d1 < d2:
            print(np.sqrt(d0)/20, np.sqrt(d1)/30)
            pixels_per_cm_data.append(np.sqrt(d0)/20)
            pixels_per_cm_data.append(np.sqrt(d0)/20)
            return perm


def find_visual_landmarks(clusters, all_circs):
    visual_landmarks = []
    for cluster in clusters:
        if len(cluster) == 3:
            visual_landmarks.append(oriented_cluster(cluster, all_circs))
    return visual_landmarks

def generate_landmark_Image(visual_landmarks, all_circs, dst_copy, show = False): 
    if all_circs is None: 
        print("No circles found")  
    else:
        all_circs_rounded = np.intc(all_circs)
        for landmark in visual_landmarks:
            circ = all_circs_rounded[0]
            x0 = circ[landmark[0]][0]
            y0 = circ[landmark[0]][1]
            x1 = circ[landmark[1]][0]
            y1 = circ[landmark[1]][1]
            x2 = circ[landmark[2]][0]
            y2 = circ[landmark[2]][1]
            deltax = x1-x0
            deltay = y1-y0
            x3 = x2 - deltax
            y3 = y2 - deltay
            #angle1 = ((np.arctan2(y1-y0, x1-x0)* 180 / np.pi)+360)%360
            #angle2 = ((np.arctan2(y2-y1, x2-x1)* 180 / np.pi)+360)%360
            #print('here')
            #print(angle1)

            #points enclosing id
            m1x = (3*x0+2*x2)/5.0
            m1y = (3*y0+2*y2)/5.0
            m2x = np.intc((3*m1x+2*x3)/5.0)
            m2y = np.intc((3*m1y+2*y3)/5.0)

            if show:
                print(deltax, deltay, x0, x1, y0, y1)
                cv.line(dst_copy, (x0,y0), (x1,y1), (0,255,0), 2)
                cv.line(dst_copy, (x1,y1), (x2,y2), (0,255,0), 2)
                cv.line(dst_copy, (x2,y2), (x3,y3), (0,255,0), 2)
                cv.line(dst_copy, (x3,y3), (x0,y0), (0,255,0), 2)
                cv.rectangle(dst_copy, (m2x-32,m2y-32),(m2x+32,m2y+32), (255,0,0),2)
                

        if show:
            for circ in all_circs_rounded[0]:
                cv.circle(dst_copy, (circ[0], circ[1]), circ[2], (0,0,255), 2)
                cv.circle(dst_copy, (circ[0], circ[1]), 0, (255,0,0), -1)
           
            plt.imshow(dst_copy)
            plt.imsave('Landmark_ID2.jpg', dst_copy)

            plt.show(block = False)

def generate_landmark_ID_Image(landmark, all_circs, dst):
    circ = all_circs[0]
    x0 = circ[landmark[0]][0]
    y0 = circ[landmark[0]][1]
    x1 = circ[landmark[1]][0]
    y1 = circ[landmark[1]][1]
    x2 = circ[landmark[2]][0]
    y2 = circ[landmark[2]][1]
    deltax = x1-x0
    deltay = y1-y0
    x3 = x2 - deltax
    y3 = y2 - deltay
    #mid points
    m1x = (3*x0+2*x2)/5.0
    m1y = (3*y0+2*y2)/5.0
    m2x = np.intc((3*m1x+2*x3)/5.0)
    m2y = np.intc((3*m1y+2*y3)/5.0)
    #angle1 = ((np.arctan2(y1-y0, x1-x0)* 180 / np.pi)+360)%360
    if x0 == x1:
        angle1 = np.pi/2
    else:
        angle1 = np.arctan((-y1+y0)/(-x1+x0))
    #print(angle1)
    if m2y-32 < 0 or m2x - 32 < 0 or m2y+32 > h or m2x+32 > w:
        return None, angle1
    return dst[m2y-32:m2y+32,m2x-32:m2x+32], angle1
 
def create_3_digit(num):
    '''create_3_digit(num) -> 3 digit string
    Assumes given numbers have 3 or less digits'''
    num = str(num)
    if len(num) == 3:
        return str(num)
    if len(num) == 2:
        return '0' + num
    if len(num) == 1:
        return '00' + num

def process_img(img):
    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    #REMOVE IF DOESN'T WORK
    #REMOVE IF DOESN'T WORK
    #REMOVE IF DOESN'T WORK
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    #REMOVE IF DOESN'T WORK
    #REMOVE IF DOESN'T WORK
    dst_copy = dst.copy()
    dst = cv.cvtColor(dst,cv.COLOR_BGR2GRAY)
    dst = cv.GaussianBlur(dst, (7,7), cv.BORDER_DEFAULT)
    all_circs = cv.HoughCircles(dst, cv.HOUGH_GRADIENT, 1, 70, param1 = 50, param2 = 30, minRadius = 45, maxRadius = 55)
    clusters = find_circle_clusters(all_circs)
    visual_landmarks = find_visual_landmarks(clusters, all_circs)
    generate_landmark_Image(visual_landmarks, all_circs, dst_copy,True)
    lID = []
    angles = []
    for landmark in visual_landmarks:
        idImg, angle = generate_landmark_ID_Image(landmark, all_circs, dst_copy)
        if idImg is not None:
            ''' Code to generate training data sets
            f = 'Landmark ID Images/ID'+create_3_digit(count)+'.jpg'
            print(f)
            cv.imwrite(f,idImg)
            '''
            #pre-process
            idImg = cv.cvtColor(idImg,cv.COLOR_BGR2GRAY)
            idImg = cv.GaussianBlur(idImg, (3,3), cv.BORDER_DEFAULT)
            X_normalized = np.array(np.array([idImg])/ 255.0 - 0.5 )
            prob = model.predict(X_normalized, verbose = 0)
            lID.append(prob.argmax() + 1)
        else:
            lID.append(None)
        angles.append(angle)
    frame_info = [all_circs, visual_landmarks, lID, angles]
    return frame_info

def align_axis(frame_info):
    heading = np.average(np.array(frame_info[3]))
    angle  = -heading
    all_circs = frame_info[0]
    vl = frame_info[1][0]
    R = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
    #all_circs_coords = all_circs[:,:,0:2][0] - np.array([w/2,h/2]) #image center coordinates
    all_circs_coords = all_circs[:,:,0:2][0] - np.array([800,600]) #image center coordinates
    all_circs_coords = np.dot(R,all_circs_coords.T).T 
    if all_circs_coords[vl[1]][1] < all_circs_coords[vl[2]][1]: # rotate another 180
        #print('rotating 180')
        angle = np.pi
        heading += angle
        R = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
        all_circs_coords = np.dot(R,all_circs_coords.T).T 
    #all_circs_coords = all_circs_coords + np.array([w/2,h/2]) #image coordinates
    all_circs_coords = all_circs_coords + np.array([800,600]) #image coordinates
    return all_circs_coords, (360+heading*180/np.pi)%360

# Generate data for reference frame (one time) and write to yaml file
'''
img = cv.imread('CalImgs2/Test8.jpg')
frame_info = process_img(img)
aligned_circs_coords, heading = align_axis(frame_info)
#transform the matrix and distortion coefficients to writable lists
data = {'all_circs':np.asarray(frame_info[0]).tolist(), 'aligned_circs_coords': np.asarray(aligned_circs_coords).tolist(),
        'landmarks':np.asarray(frame_info[1]).tolist(), 'lID':np.asarray(frame_info[2]).tolist(),
        'angles':np.asarray(frame_info[3]).tolist(), 'heading':np.asarray(heading).tolist()}

# and save it to a file
with open("reference_frame.yaml", "w") as f:
    yaml.dump(data, f)
'''
# Read reference frame info
with open("reference_frame.yaml", "r") as f:
    data = yaml.safe_load(f)
f.close()
ref_all_circs = np.array(data['all_circs'])
ref_aligned_circs = np.array(data['aligned_circs_coords'])
ref_landmarks = np.array(data['landmarks'])
ref_angles = np.array(data['angles'])
ref_lID = np.array(data['lID'])
ref_heading = data['heading']
ref_camera_coords = np.array(WaldosWorld.transform(*WaldosWorld.WP3, 0, WaldosWorld.cameraCoords, WaldosWorld.bodyCoords))

np.array([51.1+90+5,44.8+6.5+90+8]) #cms

def find_coords(aligned_circs, landmarks, lID):
    #Reference frame has all landmarks, so lID is common list
    deltax = []
    deltay = []
    sanity = True
    for i, idx in enumerate(lID):
        if idx is None:
            continue
        landmark = landmarks[i]
        ri = np.where(ref_lID==idx)[0][0]
        ref_landmark = ref_landmarks[ri]
        dx = aligned_circs[landmark[1]][0] - ref_aligned_circs[ref_landmark[1]][0]
        dy = aligned_circs[landmark[1]][1] - ref_aligned_circs[ref_landmark[1]][1]
        deltax.append(dx)
        deltay.append(dy)
    deltax = np.array(deltax)
    deltay = np.array(deltay)
    print(deltax, deltay, lID)
    if np.abs(deltax.max()-deltax.min()) > 50 or np.abs(deltay.max()-deltay.min()) > 50:
        print('sanity check failed')
        sanity = False
    camera_coords = ref_camera_coords + (np.average(deltax)/pixels_per_cm,np.average(deltay)/pixels_per_cm)
    return np.array(camera_coords), sanity

def camera_gt_images(images):
    now = time.time()
    camera_gt_data = []
    sanity_data = []
    heading_data = []
    for fname in images:
        img = cv.imread(fname)
        frame_info = process_img(img)
        aligned_circs, heading = align_axis(frame_info)
        print(frame_info)
        print(aligned_circs)
        camera_coords, sanity = find_coords(aligned_circs, frame_info[1], frame_info[2])
        camera_gt_data.append(camera_coords)
        sanity_data.append(sanity)
        heading_data.append(heading)
    return np.array(camera_gt_data), np.array(heading_data), np.array(sanity_data), time.time()-now

def camera_gt_video(fname):
    # VideoCapture object
    now = time.time()
    cap = cv.VideoCapture(fname)
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
       print("Error opening video stream or file")
    # Read until video is completed
    fcount = 0
    camera_gt_data = []
    sanity_data = []
    heading_data = []
    while(cap.isOpened()):
        #print('processing')
        #Capture frame-by-frame
        ret, img = cap.read()
        if ret == False:
            break
        print(fcount)
        
        if fcount >0:
            break
        '''
        if fcount % 5 != 0 or fcount < 14*25 or fcount > 96*25:
        #if fcount % 5 != 0 or fcount < 1180 or fcount > 63*25:
            fcount += 1
            continue
        '''
        frame_info = process_img(img)
        aligned_circs, heading = align_axis(frame_info)
        #print(frame_info)
        #print(aligned_circs)
        #print(heading)
        camera_coords, sanity = find_coords(aligned_circs, frame_info[1], frame_info[2])
        print(camera_coords)
        camera_gt_data.append(camera_coords)
        heading_data.append(heading)
        sanity_data.append(sanity)
        fcount += 1
    cap.release()
    print(camera_gt_data)
    return np.array(camera_gt_data), np.array(heading_data), np.array(sanity_data), time.time()-now
'''
fname = 'Data/HR_M_RP_W21_C1.h264'
camera_gt_data, heading_data, sanity_data, time_to_process = camera_gt_video(fname)
good_camera_gt_data = camera_gt_data[sanity_data]
good_body_gt_data = WaldosWorld.transform(good_camera_gt_data[:,0], good_camera_gt_data[:,1], 0, WaldosWorld.bodyCoords, WaldosWorld.cameraCoords)
ax = WaldosWorld.plot_RectView()
ax.scatter(good_body_gt_data[0], good_body_gt_data[1])
plt.show(block=False)
#sTIdx = camera_gt_data(
'''

#images = glob.glob('./Calibration Images/Test*.jpg')
#images = ['CalImgs2/Test1.jpg']
#images = ['CalImgs2/Test8.jpg']
#images = ['Data/HRtestWP21.jpg']
#images = ['Data/0testWP21.jpg']
#camera_gt_data, heading_data, sanity_data, time_to_process=camera_gt_images(images)
#good_camera_gt_data = camera_gt_data[sanity_data]
#good_body_gt_data = WaldosWorld.transform(good_camera_gt_data[:,0], good_camera_gt_data[:,1], 0, WaldosWorld.bodyCoords, WaldosWorld.cameraCoords)

#images = glob.glob('./CameraTest/*.jpg')
'''
images = ['./CameraTest/C_WP3_180.jpg']
for fname in images:
    img = cv.imread(fname)
    frame_info = process_img(img)
    print(fname)
    print(frame_info)
'''
s = np.array([815, 634])
WP3 = np.array([457.5, 247.5])
angle = np.pi
WP3_Shift = WP3 - s #image center coordinates
R = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
WP3_ShiftRotated = np.dot(R, WP3_Shift)
WP3_Rotated = WP3_ShiftRotated + s
