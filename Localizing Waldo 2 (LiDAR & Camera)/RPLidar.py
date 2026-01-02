import WaldosWorld
import numpy as np
import matplotlib.pyplot as plt

#def load_TFLuna(fileStr):
frp = 'Data/test_w01RP.npy'
initState = np.array([*WaldosWorld.WP1,90])
rpCoords = WaldosWorld.transform(initState[0], initState[1], initState[2], WaldosWorld.rpCoords, WaldosWorld.bodyCoords)


rp_data = np.load(frp, allow_pickle = True)

d = rp_data[0][3][:,2]/10
az = (180-rp_data[0][3][:,1] + 360)%360
az = az*np.pi/180

error = []
for i, angle in enumerate(az):
    dT = WaldosWorld.find_distance(*rpCoords,angle+np.pi/2)
    print(d[i], dT)
    error.append(d[i]-dT)

fig, ax = plt.subplots(subplot_kw = {'projection': 'polar'})
ax.scatter(az, error, marker = '.', c='crimson')
ax.set_title('RPLiDAR A1 Scan Error, WP1, 90 Heading', fontsize = 10)
plt.show(block = False)
plt.savefig('RPErrorWP3.jpg')

ax1 = WaldosWorld.plot_LiDARView(*rpCoords, initState[2], 'RPLiDAR A1 Scan, WP1, 90 Heading')
ax1.scatter(az, d, marker ='.', c = 'crimson')
plt.show(block=False)
plt.savefig('RPWP3.jpg')

print(len(error), np.average(error), np.std(error))


