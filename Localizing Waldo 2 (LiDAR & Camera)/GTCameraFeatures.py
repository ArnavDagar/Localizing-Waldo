import cv2 as cv
import numpy as np
import yaml
import matplotlib.pyplot as plt

# Read camera matrix
with open("callibration_matrix.yaml", "r") as f:
    data = yaml.safe_load(f)
f.close()
mtx = np.array(data['camera_matrix'])
dist = np.array(data['dist_coeff'])

(h, w) = (720, 1280)
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))


fname = 'Data/HighResVid.h264'

frameCount = 0

x_init = 280
y_init = 280
theta_init = np.pi

sift = cv.SIFT_create()
# VideoCapture object 
cap = cv.VideoCapture(fname)
cap.get(cv2.CAP_PROP_FPS)
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")
 
# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        frameCount += 1
        print(frameCount)
        if frameCount == 1:
            print(frame.shape)
            fig1, ax1 = plt.subplots()

            ax1.imshow(frame)
           
            plt.show(block=False)
            break
            # undistort
            frame1 = cv.undistort(frame, mtx, dist, None, newcameramtx)
            print(frame1.shape)

        elif frameCount == 671:
            frame2 = cv.undistort(frame, mtx, dist, None, newcameramtx)

            # find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(frame1,None)
            kp2, des2 = sift.detectAndCompute(frame2,None)

            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)
            flann = cv.FlannBasedMatcher(index_params,search_params)
            matches = flann.knnMatch(des1,des2,k=2)
            pts1 = []
            pts2 = []

            # ratio test as per Lowe's paper
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.8*n.distance:
                    pts2.append(kp2[m.trainIdx].pt)
                    pts1.append(kp1[m.queryIdx].pt)

            pts1 = np.int32(pts1)
            pts2 = np.int32(pts2)

            pts1 = np.array([[163, 134], [252,134], [163, 221], [252, 222], [268, 18],[431, 232], [448,440],[372,87] ])
            pts2 = np.array([[967, 124], [1054,122], [968, 212], [1056, 209], [1064, 11], [1244,211], [1249,420], [1166,73]])

            E = cv.findEssentialMat(pts1, pts2, mtx)
            R1, R2, t = cv.decomposeEssentialMat(E[0])

            H, status = cv.findHomography(pts1, pts2)
            H = np.transpose(H)
            h1 = H[0]
            h2 = H[1]
            h3 = H[2]

            Ainv = np.linalg.inv(mtx)

            L = 1 / np.linalg.norm(np.dot(Ainv, h1))

            r1 = L * np.dot(Ainv, h1)
            r2 = L * np.dot(Ainv, h2)
            r3 = np.cross(r1, r2)

            T = L * np.dot(Ainv, h3)

            R = np.array([[r1], [r2], [r3]])
            R = np.reshape(R, (3, 3))
            U, S, V = np.linalg.svd(R, full_matrices=True)

            U = np.matrix(U)
            V = np.matrix(V)
            R = U * V

            '''
            H = H.T
            h1 = H[0]
            h2 = H[1]
            h3 = H[2]
            K_inv = np.linalg.inv(mtx)
            L = 1 / np.linalg.norm(np.dot(K_inv, h1))
            r1 = L * np.dot(K_inv, h1)
            r2 = L * np.dot(K_inv, h2)
            r3 = np.cross(r1, r2)
            T = L * (K_inv @ h3.reshape(3, 1))
            R = np.array([[r1], [r2], [r3]])
            R = np.reshape(R, (3, 3))
            '''
            #frame1 = frame2
        # Break the loop
    else:
        break
 
# Release the video capture object
print('here')
cap.release()
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
ax1.imshow(frame1)
ax2.imshow(frame2)
plt.show(block=False)
 
