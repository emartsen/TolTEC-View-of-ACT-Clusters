from matplotlib import pyplot as plt
from ClusterUPP import ClusterUPP
from TolTEC import TolTEC
import numpy as np
plt.ion()

radius = 3.
time = 1.
mapAreaArcmin2 = np.pi*radius**2
mapAreaDeg2 = mapAreaArcmin2/3600.

T = TolTEC(2.0)
depth_mJy = T.depth_mJy(mapAreaDeg2, time)
depth_y = T.depth_y(mapAreaArcmin2, time)

#convert y to T
c = ClusterUPP(4.e14, 0.5)
Tfromy = c.deltaT(depth_y,T.nu*1.e-9)
Tpfromy = T.y2K(depth_y)

#convert S to T
TfromS = T.Jy2K(depth_mJy*1.e-3)


print("depth in mJy: {0:3.2f}".format(depth_mJy))
print("depth in y: {0:3.2e}".format(depth_y))
print("")
print("depth in deltaT from y(T): {0:3.2e}".format(Tfromy))
print("depth in deltaT from Jy2K: {0:3.2e}".format(TfromS))
print("depth in T: {0:3.2e}".format(Tpfromy))


# nat edit: to plot dI/I
plot = 1

if plot:
    vCl = 1000 # km/s
    M = 3e14 # solar mass
    z = 0.7 #redshift
    y = 5e-4
    
    c1 = ClusterUPP(M, z, vP = vCl)
    
    nu = np.arange(0.0001, 500)
    c1.plotSZE(y, nu, c1.theta500arcmin) 
