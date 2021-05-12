
# this code computes the seeing measurements based on measurements done with a DIMM mask.
# it takes all images from a certain folder to compute the seeing.
# in the end you get a dataframe (vardf) with the calculated variances in x, y, radial and angle, and the seeing at zenith. 

# importing needed libraries
import numpy as np
from astropy.io import fits
import os
import pandas as pd
from matplotlib import pyplot as plt
import photutils
import time

start_time = time.time()
path = r'D:\Master\ZWO\trial\2021-04-11_20_44_39Z'
os.chdir(path)
darki = fits.open(r'D:\Master\ZWO\2021-04-12_19_27_48Z_dark\dark.fit')
dark = darki[0].data
pscale = 0.604
diffy = []
diffx = []
for image in os.listdir(path):
    hdul = fits.open(image)  # Open image
    data = hdul[0].data  # data = pixel brightness  
    data = data.astype(float)
    data -= dark.astype(float)
    data = data/np.std(data)
    plt.close()
    plt.imshow(data)
    plt.colorbar()
    
    w = np.where(data >= 0.6*np.max(data)) # Brightest pixel position (y,x)
    hb = 150 # half box size around w 
    yc = int(w[0][0]) # index of centroid row 
    xc = int(w[1][0]) # index of centroid column 
    if yc-hb <0 or yc+hb> len(data[0]) or xc-hb<0 or xc+hb > len(data[1]) :
        print('the location of the star is less than %i pixels from the edge' %hb)
    window = data[(yc-hb):(yc+hb), (xc-hb):(xc+hb)] # box limits: data[yc+-hb, xc+-hb]
    plt.close()
    plt.imshow(window)
    plt.colorbar()

    threshold = 12 # threshold value for star detection (in SNR)
    fwhm = 6  # estimated fwhm of the star 
    IRAF = photutils.detection.DAOStarFinder(threshold,fwhm) 
    T = IRAF.find_stars(window)
    print(T)
    if T == None or len(T) > 2 or len(T) < 2: 
        print('more or less than 2 stars detected!')
        continue

    diffy.append(np.abs(T['ycentroid'][0] - T['ycentroid'][1])*pscale)
    diffx.append(np.abs(T['xcentroid'][0] - T['xcentroid'][1])*pscale)

    TIME = pd.to_datetime(hdul[0].header['DATE-OBS'][:10] +' '+ hdul[0].header['DATE-OBS'][11:],format = '%Y-%m-%d %H:%M:%S')



print("--- %.2f seconds ---" % (time.time()-start_time))


"""
# find centroids of the stars 
cents = np.where(window >= 6) # position of pixels with SNR >= value in "window" 
cents = np.where(likel >= 10)

s1y = []
s2y = []
s1x = []
s2x = []
for y in range(len(cents[0])-1):
    if abs(cents[0][y] - cents[0][y+1]) > 10: # if the distance between the pixels is >3 
        break 
    s1y = cents[0][:y]
    s2y = cents[0][(y+1):]

for x in range(len(cents[1])-1):
    if abs(cents[1][x] - cents[1][x+1]) > 10:
        break
    s1x = cents[1][:x]
    s2x = cents[1][(x+1):]

hb1 = 15 # half box size around w 
s1y = int(np.median(s1y)) # index of centroid row of star 1 
s1x = int(np.median(s1x)) # index of centroid column  of star 1 
w1 = window[(s1y-hb1):(s1y+hb1), (s1x-hb1):(s1x+hb1)] 
plt.imshow(w1)
cen1 = np.where(w1 == np.max(w1))

s2y = int(np.median(s2y)) # index of centroid row of star 1 
s2x = int(np.median(s2x)) # index of centroid column  of star 1 
w2 = window[(s2y-hb1):(s2y+hb1), (s2x-hb1):(s2x+hb1)] 
plt.imshow(w2)
cen2 = np.where(w2 == np.max(w2))

wcen1 = s1y-hb1+cen1[0], s1x-hb1+cen1[1]  # star 1 centroid positions in "window" 
wcen2 = s2y-hb1+cen2[0], s2x-hb1+cen2[1] # star 2 centroid positions in "window"
plt.imshow(window) # visualize to see the distance
"""