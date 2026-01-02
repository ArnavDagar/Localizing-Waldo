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

count = 3133

# Load reference data for each of the 36 images at refA and refB
# These were processed once and stored for future use

refA = np.load('refA.npy', allow_pickle = True)
refB = np.load('refB.npy', allow_pickle = True)
refACoords =WaldosWorld.refACoords
refBCoords =WaldosWorld.refBCoords


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

def normalize_image(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized - it expects values between 0,255
    :return: Normalized image data
    """
    a = 0.1
    b = 0.9
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )

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
            #print(np.sqrt(d0)/20, np.sqrt(d1)/30)
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
                '''
                print(deltax, deltay, x0, x1, y0, y1)
                cv.line(dst_copy, (x0,y0), (x1,y1), (0,255,0), 2)
                cv.line(dst_copy, (x1,y1), (x2,y2), (0,255,0), 2)
                cv.line(dst_copy, (x2,y2), (x3,y3), (0,255,0), 2)
                cv.line(dst_copy, (x3,y3), (x0,y0), (0,255,0), 2)
                if not(m2y-32 < 0 or m2x - 32 < 0 or m2y+32 > h or m2x+32 > w or x3 < 0 or x3 > w or y3 < 0 or y3 > h):
                    cv.rectangle(dst_copy, (m2x-32,m2y-32),(m2x+32,m2y+32), (255,0,0),2)
                '''

        if show:
            '''
            for circ in all_circs_rounded[0]:
                cv.circle(dst_copy, (circ[0], circ[1]), circ[2], (0,0,255), 2)
                cv.circle(dst_copy, (circ[0], circ[1]), 0, (255,0,0), -1)
            '''
            plt.imshow(dst_copy)
            plt.imsave('UndistortedTest1.jpg', dst_copy)

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
    
    #if np.abs(x0-x1)<=10:
    if x0==x1:
        angle1 = np.pi/2
    else:
        # Be careful, coordinate system is right handed
        #print(x1-x0)
        y1 = -y1
        y0 = -y0
        angle1 = np.arctan((-y1+y0)/(-x1+x0))
    print('angle rads is '+ str(angle1))
    if m2y-32 < 0 or m2x - 32 < 0 or m2y+32 > h or m2x+32 > w or x3 < 0 or x3 > w or y3 < 0 or y3 > h:
        return None, angle1
    return dst[m2y-32:m2y+32,m2x-32:m2x+32], angle1
 
def create_4_digit(num):
    '''create_4_digit(num) -> 4 digit string
    Assumes given numbers have 4 or less digits'''
    num = str(num)
    if len(num) == 4:
        return num
    if len(num) == 3:
        return '0' + num
    if len(num) == 2:
        return '00' + num
    if len(num) == 1:
        return '000' + num

def process_img(img):
    global count
    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
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
            #Code to generate training data sets
            '''
            f = 'LandmarkIDImages/ID'+create_4_digit(count)+'.jpg'
            print(f)
            cv.imwrite(f,idImg)
            count += 1
            continue
            '''
            #pre-process
            s = time.time()
            idImg = cv.cvtColor(idImg,cv.COLOR_BGR2GRAY)
            X_normalized = np.array([normalize_image(idImg)])
            prob = model.predict(X_normalized, verbose = 0)
            lID.append(prob.argmax() + 1)
        else:
            lID.append(-1)
        angles.append(angle)
    frame_info = [all_circs, visual_landmarks, lID, angles]
    return frame_info

def rotate(frame_info, angle):
    all_circs = frame_info[0]
    R = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
    all_circs_coords = all_circs[:,:,0:2][0]
    #print(all_circs_coords)
    all_circs_coords = all_circs_coords - np.array([w/2,h/2]) #image center coordinates
    all_circs_coords[:,1] = -all_circs_coords[:,1] # make coordinate system left handed
    all_circs_coords = np.dot(R,all_circs_coords.T).T
    all_circs_coords[:,1] = -all_circs_coords[:,1] # make coordinate system aligned with image
    all_circs_coords = all_circs_coords + np.array([w/2,h/2]) #image coordinates
    return all_circs_coords
    
    
def align_axis(frame_info):
    #heading = np.average(np.array(frame_info[3]))
    heading = frame_info[3][0]
    angle  = -heading
    all_circs = frame_info[0]
    vl = frame_info[1][0]
    R = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
    all_circs_coords = all_circs[:,:,0:2][0]
    #print(all_circs_coords)
    all_circs_coords = all_circs_coords - np.array([w/2,h/2]) #image center coordinates
    all_circs_coords[:,1] = -all_circs_coords[:,1] # make coordinate system left handed
    all_circs_coords = np.dot(R,all_circs_coords.T).T
    if all_circs_coords[vl[1]][1] > all_circs_coords[vl[2]][1]: # rotate another 180
        #print('rotating 180')
        angle = np.pi
        heading += angle
        R = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
        #print(all_circs_coords)
        all_circs_coords = np.dot(R,all_circs_coords.T).T 
    all_circs_coords[:,1] = -all_circs_coords[:,1] # make coordinate system aligned with image
    all_circs_coords = all_circs_coords + np.array([w/2,h/2]) #image coordinates
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


def calc_coords(frame_info, heading):
    circs, landmarks, lID, angles = frame_info
    #print(circs, landmarks, lID, heading)
    if 2 in lID or 3 in lID:
        ref = refA
        refCoords = refACoords
    elif 1 in lID or 4 in lID:
        ref = refB
        refCoords = refBCoords
    rem = heading%10
    if rem > 5:
        idx = (np.intc(heading//10)+1)%36
    else:
        idx = np.intc(heading//10)
    #print('idx')
    #print(idx)
    #print('refCoords')
    #print(refCoords)
    ref_camera_coords = WaldosWorld.transform(*refCoords, idx*np.pi/18.0, WaldosWorld.cameraCoords, WaldosWorld.bodyCoords)
    #print('ref_camera_coords')
    #print(ref_camera_coords)
    refcircs, reflandmarks, reflID, refangles = ref[idx][0]
    #print(refcircs, reflandmarks, reflID, refangles)
    
    circs = np.array([rotate(frame_info, (ref[idx][-1] - heading)*np.pi/180)])
    #print('rotated circs '+ str(ref[idx][-1] - heading) )
    #print(circs)
    
    matchIDs, idx1, idx2 = np.intersect1d(lID, reflID, return_indices = True)
    if len(matchIDs) == 0:
        print('No match found!?!')
        sanity = False
        return np.array(ref_camera_coords), sanity
    if len(matchIDs) == 1:
        print('Only one match found!?!')
        sanity = False
    #print(matchIDs, idx1, idx2, lID, reflID)
    angle = ref[idx][-1]*np.pi/180.0
    R = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
    deltax = []
    deltay = []
    sanity = True
    for i, id in enumerate(matchIDs):
        if id is None:
            continue
        landmark = landmarks[idx1[i]]

        ref_landmark = reflandmarks[idx2[i]]
        #print(landmark, ref_landmark)
        dx = circs[0][landmark[1]][0] - refcircs[0][ref_landmark[1]][0]
        dy = circs[0][landmark[1]][1] - refcircs[0][ref_landmark[1]][1]
        
        t = np.dot(R,np.array([dx,dy]))
        deltax.append(t[0])
        deltay.append(t[1])
    deltax = np.array(deltax)
    deltay = np.array(deltay)
    if np.abs(deltax.max()-deltax.min()) > 50 or np.abs(deltay.max()-deltay.min()) > 50:
        print('sanity check failed')
        sanity = False
    #print(deltax, deltay)
    camera_coords = ref_camera_coords + np.array([np.average(deltax)/pixels_per_cm,np.average(deltay)/pixels_per_cm])
    #print(camera_coords, ref_camera_coords)
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
        print('info')
        print(frame_info)
        print(aligned_circs)
        print('heading')
        '''
        heading1 = (heading//10)*10
        camera_coords1, sanity1 = calc_coords(frame_info, heading1)
        heading2 = ((heading//10)*10+10)%360
        camera_coords2, sanity2 = calc_coords(frame_info, heading2)
        if heading >= 350:
            scale = (heading - 350)/10
        else:
            scale = (heading-heading1)/10
        camera_coords = scale*camera_coords1 + (1-scale)*camera_coords2
        sanity = sanity1 & sanity2
        print(camera_coords, camera_coords1, camera_coords2)
        '''
        camera_coords, sanity = calc_coords(frame_info, heading)
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
        print('framecount '+ str(fcount))

        '''
        if fcount>400 :
            break
            fcount += 1
            continue
        '''
        
        if fcount % 5 == 1 or fcount % 5 == 3 or fcount % 5 == 4:
            fcount += 1
            continue
        
        frame_info = process_img(img)

        aligned_circs, heading = align_axis(frame_info)
        #print(frame_info)
        #print(aligned_circs)
        #print(heading)
        #camera_coords, sanity = find_coords(aligned_circs, frame_info[1], frame_info[2])
        camera_coords, sanity = calc_coords(frame_info, heading)

        '''
        if not sanity:
            break
        '''

        #print(camera_coords)
        camera_gt_data.append(camera_coords)
        heading_data.append(heading)
        sanity_data.append(sanity)
        fcount += 1
    cap.release()
    #print(camera_gt_data)
    return np.array(camera_gt_data), np.array(heading_data), np.array(sanity_data), time.time()-now

'''
#fname = 'Data/Video/M_RP_W15_O270_C_2.h264'
fname = 'Data/Video/HR_M_RP_W21_S1.h264'
camera_gt_data, heading_data, sanity_data, time_to_process = camera_gt_video(fname)
good_camera_gt_data = camera_gt_data[sanity_data]
good_body_gt_data = WaldosWorld.transform(good_camera_gt_data[:,0], good_camera_gt_data[:,1], np.pi*heading_data[sanity_data]/180, WaldosWorld.bodyCoords, WaldosWorld.cameraCoords)
ax = WaldosWorld.plot_RectView('Camera Localization')
ax.plot(good_body_gt_data[0], good_body_gt_data[1])
plt.show(block=False)
plt.savefig('CameraS1.jpg')
#sTIdx = camera_gt_data(
'''

'''
images = []
for i in range(145):
    fname = 'CamLoc/Test{}.jpg'.format(i)
    images.append(fname)
#images = glob.glob('CamLoc/Test*.jpg')
#images = ['CamLoc/Test10.jpg', 'CamLoc/Test11.jpg', 'CamLoc/Test12.jpg', 'CamLoc/Test13.jpg', 'CamLoc/Test14.jpg']
#images = ['CamLoc/Test135.jpg','CamLoc/Test136.jpg','CamLoc/Test137.jpg','CamLoc/Test138.jpg','CamLoc/Test139.jpg']
#images = ['CalImgs2/Test1.jpg']
#images = ['CalImgs2/Test8.jpg']
#images = ['Data/HRtestWP21.jpg']
#images = ['Data/0testWP21.jpg']
camera_gt_data, heading_data, sanity_data, time_to_process=camera_gt_images(images)
sanity_data = np.array([True]*len(sanity_data))
good_camera_gt_data = camera_gt_data[sanity_data]
good_body_gt_data = WaldosWorld.transform(good_camera_gt_data[:,0], good_camera_gt_data[:,1], heading_data[sanity_data], WaldosWorld.bodyCoords, WaldosWorld.cameraCoords)
ax = WaldosWorld.plot_RectView('Camera Localization')
'''
'''
for i in range(18):
    #ax.plot(good_body_gt_data[0][5*i:5*i+5], good_body_gt_data[1][5*i:5*i+5], c = 'crimson')
    ax.scatter(good_body_gt_data[0][5*i:5*i+5], good_body_gt_data[1][5*i:5*i+5], c = 'crimson', marker = '.')
'''
'''
gtX = np.repeat(WaldosWorld.wpCoords[:,0],5)
gtY = np.repeat(WaldosWorld.wpCoords[:,1],5)
errX = good_camera_gt_data[:,0]-gtX
errY = good_camera_gt_data[:,1]-gtY
err = np.sqrt(errX**2, errY**2)
bins = np.arange(0,30,0.25)
count, bins_count = np.histogram(err, bins = bins)
pdf = count/sum(count)
cdf = np.cumsum(pdf)

fig, ax = plt.subplots()
ax.set_xlabel('cm')
ax.set_ylabel('Probability <=')
ax.plot(bins[1:], cdf)
ax.set_title('CDF Of Camera Localization Errors', fontsize = 10)
    
plt.show(block=False)
plt.savefig('CamScatter.jpg')
'''

images = ['Calibration Images/Test0.jpg']

camera_gt_data, heading_data, sanity_data, time_to_process = camera_gt_images(images)
'''
good_camera_gt_data = camera_gt_data[sanity_data]
good_body_gt_data = WaldosWorld.transform(good_camera_gt_data[:,0], good_camera_gt_data[:,1], 0, WaldosWorld.bodyCoords, WaldosWorld.cameraCoords)


refB = []
for i in range(36):
    fname = 'CalImgsA/Test{}.jpg'.format(i)
    print(fname)
    img = cv.imread(fname)
    frame_info = process_img(img)
    aligned_circs, heading = align_axis(frame_info)
    refB.append([frame_info,aligned_circs,heading])
np.save('refA.npy', np.array(refB, dtype = object))

fname = 'CalImgsA/Test{}.jpg'.format(2)
print(fname)
img = cv.imread(fname)
frame_info = process_img(img)
aligned_circs, heading = align_axis(frame_info)
'''
