# importing needed libraries
import numpy as np
from astropy.io import fits
import os
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mp 
from scipy import stats as st
import datetime as dt
from datetime import datetime, timedelta

path = r'D:\Master\ZWO\trial\2021-04-11_20_44_39Z'
os.chdir(path)
darki = fits.open(r'D:\Master\ZWO\2021-04-12_19_27_48Z_dark\dark.fit')
dark = darki[0].data
pscale = 0.604

##### functions 
diffy = []
diffx = []
radial = [] 
angle = [] 
Date = []
Hour = []
Time = [] 
for image in os.listdir(path):
    os.chdir(path)  # define the path as current directory
    hdul = fits.open(image)  # Open image
    data = hdul[0].data  # data = pixel brightness  
    data = data.astype(float)
    data -= dark.astype(float)
    #data -= np.median(data)
    data = data/np.std(data)
    background = abs(np.median(data))
    
    w = np.where(data >= 15*background) # Brightest pixels position (y,x)
    #w = np.where(data >= 0.9*np.max(data)) 
    hb = 150 # half box size around w 
    yc = int(w[0][0]) # index of centroid row 
    xc = int(w[1][0]) # index of centroid column 
    if yc-hb <0 or yc+hb > len(data[0]) or xc-hb<0 or xc+hb > len(data[1]):
        print('\nThe location of the star is less than %i pixels from the edge. \nThis image was discarded: %s' %(hb, image))
        continue
    window = data[(yc-hb):(yc+hb), (xc-hb):(xc+hb)] # box limits: data[yc+-hb, xc+-hb]
    plt.imshow(window) 
    
    ########## find the 2 stars (about) in window 
    cents = np.where(window >= 6*abs(np.median(window))) # position of pixels with SNR >= value in "window" 

    def cluster(array, maxdiff):
        tmp = array.copy()
        groups = []
        while len(tmp):
            # select seed
            seed = tmp.min()
            mask = (tmp - seed) <= maxdiff
            groups.append(tmp[mask, None])
            tmp = tmp[~mask]
        return groups

    starsy = [i for i in cluster(cents[0], 20) if len(i) > 10 ]
    starsx = [i for i in cluster(cents[1], 20) if len(i) > 10 ]
    
    if len (starsy) < 2:
        print("\nOnly one star found in window - this image was discarded:\n", image)
        continue
    
    ########### find centroids of stars using weighted average 
    #### centroid of star 1 
    br1 = (np.max(starsy[0]),np.max(starsx[0])) # Brightest pixel position (y,x)
    r = 3  # radius of circle area to calculate weighted mean inside 
    xw = []
    X = [] 
    yw = [] 
    Y = []
    for y in range(len(window[0])):
        for x in range(len(window[1])):
            if abs((x-br1[1])**2 + (y-br1[0])**2 - r^2) < 18:
                #window[x,y] == 30   # to visualize the location of area that ios calculated 
                xw.append(window[y,x]) 
                yw.append(window[y,x])
                X.append(x)
                Y.append(y)       
    wmean= pd.DataFrame({'xpos': X, 'xw': xw, 'ypos': Y, 'yw': yw })
    x_av1 = (wmean['xpos']*wmean['xw']).sum()/wmean['xw'].sum()  # center of mass position x axis  
    y_av1 = (wmean['ypos']*wmean['yw']).sum()/wmean['yw'].sum()  # center of mass position y axis 
    #x1.append(x_av1)
    #y1.append(y_av1)
    
    #### centroid of star 2 
    br2 = (np.max(starsy[1]),np.max(starsx[1])) # Brightest pixel position (y,x)     
    xw2 = []
    X2 = [] 
    yw2 = [] 
    Y2 = []
    for y in range(len(window[0])):
        for x in range(len(window[1])):
            if abs((x-br2[1])**2 + (y-br2[0])**2 - r^2) < 18:
                    #window[x,y] == 30   # to visualize the location of area that ios calculated 
                    xw2.append(window[y,x]) 
                    yw2.append(window[y,x])
                    X2.append(x)
                    Y2.append(y)       
    
    wmean2= pd.DataFrame({'xpos': X2, 'xw': xw2, 'ypos': Y2, 'yw': yw2 })
    x_av2 = (wmean2['xpos']*wmean2['xw']).sum()/wmean2['xw'].sum()  # center of mass position x axis 
    y_av2 = (wmean2['ypos']*wmean2['yw']).sum()/wmean2['yw'].sum()  # center of mass position y axis 
      
    dify = abs(y_av1 - y_av2)*pscale
    difx = abs(x_av1 - x_av2)*pscale
    r = np.sqrt(dify**2 + difx**2)
    ang = np.arctan(dify/difx)

    angle.append(ang)
    radial.append(r)
    diffy.append(dify)
    diffx.append(difx)
    time = pd.to_datetime(hdul[0].header['DATE-OBS'][:10] +' '+ hdul[0].header['DATE-OBS'][11:],format = '%Y-%m-%d %H:%M:%S')
    Time.append(time)
    Date.append(time.date())
    Hour.append(time.time())
    daf = {'Time':Time, 'Date': Date, 'Hour': Hour, 'diffy': diffy, 'diffx': diffx, 'radial': radial, 'angle': angle }
    df = pd.DataFrame(daf).set_index('Time') 