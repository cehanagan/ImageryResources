# -121.04097, 36.45248 to -121.04114, 36.45231
# T2 (fault perp part): -121.04082, 36.45243 to -121.04096, 36.45234
# T1: -121.04069, 36.45226 to -121.04080, 36.45217
import rioxarray
import pygmt
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Proj
P = Proj('epsg:32610')

def projectParPerp(ew,ns,az):
    # Rotation from en (xy) to fault parallel and perp, must convert azimuth
    theta = (90-az)*np.pi/180
    par = ns*np.sin(theta)+ew*np.cos(theta)
    perp = -1*ns*np.cos(theta)+ew*np.sin(theta)
    return np.array([par, perp])

file = '/Users/chanagan/Library/CloudStorage/OneDrive-DOI/Projects/CSAF_Lidar/TopoCreek/MEC_TC/ParallelDisp.tif'
par = rioxarray.open_rasterio(file)

par = par.rio.reproject('EPSG:4326')

T3 = pygmt.grdtrack(par,profile=f'-121.04097/36.45248/-121.04114/36.45231+i0.5e',
crossprofile='4e/0.5e')
T2 = pygmt.grdtrack(par,profile=f'-121.04082/36.45243/-121.04096/36.45234+i0.5e',
crossprofile='4e/0.5e')
T1 = pygmt.grdtrack(par,profile=f'-121.04069/36.45226/-121.04080/36.45217+i0.5e',
crossprofile='4e/0.5e')

for T, title in zip([T3, T2, T1],['T3', 'T2', 'T1']):
    # For each cross profile, take 2 before and 2 after for average
    profmeans = list(map(lambda i: T[((T.index >= i-2) & (T.index <= i+2))][4].mean(),T[T[2] == 0].index))
    profstd = list(map(lambda i: T[((T.index >= i-2) & (T.index <= i+2))][4].std(),T[T[2] == 0].index))
    profmeans, profstd = np.array(profmeans), np.array(profstd)
    proflons,proflats = P(T[T[2] == 0][0].values,T[T[2] == 0][1].values)
    dists = list(map(projectParPerp,proflons,proflats,[T[3][0]]*len(proflons)))
    dists = np.array(dists) - dists[-1]

    plt.figure(figsize=(6,3))
    plt.title(f'{title} profile with standard deviation from 9 perpendicular pts')

    plt.plot(dists[:,1], profmeans)
    plt.fill_between(dists[:,1], profmeans - profstd, profmeans + profstd, alpha=0.2)

    plt.xlabel('Distance along profile (m)')
    plt.ylabel('Displacement (m)')