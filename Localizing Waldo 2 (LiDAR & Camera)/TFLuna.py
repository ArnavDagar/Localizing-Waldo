import WaldosWorld
import numpy as np
import matplotlib.pyplot as plt

#def load_TFLuna(fileStr):
ftf = 'Data/S_TF_W03_O0_P_1.npy'
ftt = 'Data/S_TF_W03_O0_P_1TTMotors.npy'
tf_data = np.load(ftf, allow_pickle = True)
tt_data = np.load(ftt, allow_pickle = True)
initState = np.array([*WaldosWorld.WP3,0.0])
tflunaCoords = WaldosWorld.transform(initState[0], initState[1], initState[2], WaldosWorld.lunaCoords, WaldosWorld.bodyCoords)

start = tf_data[2,0]
end = tf_data[2,1]
scan = np.array(tf_data[2,2])
d = scan[:,2]*100
az = (scan[:,1]-start)*np.pi*2/(end-start)
lum = scan[:,3]
error = []

good = lum > 4000

for angle in az:
    dT = WaldosWorld.find_distance(*tflunaCoords,angle)
    error = d-dT

fig, ax = plt.subplots(subplot_kw = {'projection': 'polar'})
#ax.scatter(az[good], error[good], marker = '.', c='crimson')
ax.scatter(az, error, marker = '.', c='crimson')
#ax.set_title('TF Luna Scan Error, Filtered, WP3, 0 Heading', fontsize = 10)
ax.set_title('TF Luna Scan Error, WP3, 0 Heading', fontsize = 10)
plt.show(block = False)
plt.savefig('TFErrorWP3.jpg')

#ax1 = WaldosWorld.plot_LiDARView(*tflunaCoords, initState[2], 'TFLuna Scan, Filtered, WP3, 0 Heading')
ax1 = WaldosWorld.plot_LiDARView(*tflunaCoords, initState[2], 'TFLuna Scan, WP3, 0 Heading')
#ax1.scatter(az[good], d[good], marker ='.', c = 'crimson')
ax1.scatter(az, d, marker ='.', c = 'crimson')
plt.show(block=False)
plt.savefig('TFWP3.jpg')

#print(len(error[good]), np.average(error[good]), np.std(error[good]))
print(len(error), np.average(error), np.std(error))


