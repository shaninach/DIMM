#from astropy.stats import circmoment
from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt  
import pandas as pd   
#from scipy.stats import moment 

image = r'D:\Master\ZWO\trial\2021-04-11_20_44_39Z\Light_ASIImg_0.02sec_Bin1_20.1C_gain179_2021-04-11_204445_frame0006.fit'
darki = fits.open(r'D:\Master\ZWO\2021-04-12_19_27_48Z_dark\dark.fit')
dark = darki[0].data

hdul = fits.open(image)  # Open image
data = hdul[0].data  # data = pixel brightness  
data = data.astype(float)
data -= dark.astype(float)
data = data/np.std(data)
plt.close()
plt.imshow(data)
plt.colorbar()
    
w = np.where(data >= 0.6*np.max(data)) # Brightest pixel position (y,x)
hb = 100 # half box size around w 
yc = int(w[0][0]) # index of centroid row 
xc = int(w[1][0]) # index of centroid column 
if yc-hb <0 or yc+hb> len(data[0]) or xc-hb<0 or xc+hb > len(data[1]) :
    print('the location of the star is less than %i pixels from the edge' %hb)
window = data[(yc-hb):(yc+hb), (xc-hb):(xc+hb)] # box limits: data[yc+-hb, xc+-hb]
plt.close()
plt.imshow(window)
plt.colorbar()

###### weighted average to find star centroid 
# border line for taking pixels into account:
br = np.where(window == np.max(window)) # Brightest pixel position (y,x)
r = 3
k = br[0] # mean y axis 
h = br[1] # mean x axis

xw = []
X = [] 
yw = [] 
Y = []
for y in range(len(window[0])):
    for x in range(len(window[1])):
        if abs((x-h)**2 + (y-k)**2 - r^2) < 18:
            xw.append(window[x,y]) 
            yw.append(window[x,y])
            X.append(x)
            Y.append(y)
plt.imshow(window)

wmean= pd.DataFrame({'xpos': X, 'xw': xw, 'ypos': Y, 'yw': yw, })
x_av = (wmean['xpos']*wmean['xw']).sum()/wmean['xw'].sum()  # center of mass position x axis 
y_av = (wmean['ypos']*wmean['yw']).sum()/wmean['yw'].sum()  # center of mass position y axis 



