import cv2 as cv
import numpy as np
import yaml
import matplotlib.pyplot as plt

with open("calibration_matrix.yaml", "r") as f:
    data = yaml.safe_load(f)
    f.close()
mtx = np.array(data['camera_matrix'])
dist = np.array(data['dist_coeff'])

(h, w) = (1232, 1640)
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

pts1 =np.array([[643.5, 506.5],[644.5, 611.5],[803.5, 611.5],[216.5,  68.5],[ 217.5, 178.5], [381.5, 179.5 ]] )

pts1[:,0] = pts1[:,0]/w
pts1[:,1] = pts1[:,1]/h

pts2 = np.array([[ 820.5, 1010.5],[ 822.5, 1118.5],[ 983.5, 1115.5 ],[ 394.5,  588.5],[ 396.5,  694.5], [ 557.5,  691.5]])
pts2[:,0] = pts2[:,0]/w
pts2[:,1] = pts2[:,1]/h

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
