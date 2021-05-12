# this code computes the seeing measurements based on measurements done with a DIMM mask.
# it takes all images from a certain path and opens each folder in it to compute the seeing.
# in the end you get a dataframe (vardf) with the calculated variances in x, y, radial and angle, and the seeing at zenith. 

# importing needed libraries
import numpy as np
from astropy.io import fits
import os
import pandas as pd
from matplotlib import pyplot as plt
import photutils
import matplotlib as mp 
from scipy import stats as st
import datetime as dt
from datetime import datetime, timedelta
import time

start_time = time.time()

path = r'D:\Master\ZWO\trial'
os.chdir(path)

##### functions 

def var (path, dark, threshold, fwhm, pscale):
    """
    input:
    path = path to images folder
    dark = data of dark image
    threshold = minimum SNR to detect stars with IRAF
    fwhm = erstimated FWHM of stars in the image (also for IRAF)
    pscale = plate scale of telescope (arcsec/pixel)
    
    returns: dataframe with stars locations differences on y and x axes
    """
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

        w = np.where(data >= 0.7*np.max(data)) # Brightest pixel position (y,x)
        hb = 150 # half box size around w 
        yc = int(w[0][0]) # index of centroid row 
        xc = int(w[1][0]) # index of centroid column 
        if yc-hb <0 or yc+hb> len(data[0]) or xc-hb<0 or xc+hb > len(data[1]):
            print('the location of the star is less than %i pixels from the edge' %hb)
            continue
        window = data[(yc-hb):(yc+hb), (xc-hb):(xc+hb)] # box limits: data[yc+-hb, xc+-hb]
        IRAF = photutils.detection.DAOStarFinder(threshold,fwhm) 
        T = IRAF.find_stars(window)
        print(T)
        if  T == None or len(T) > 2 or len(T) < 2: 
            print('more or less than 2 stars detected!')
            continue
            
        dify = (np.abs(T['ycentroid'][0] - T['ycentroid'][1]))*pscale
        difx = (np.abs(T['xcentroid'][0] - T['xcentroid'][1]))*pscale
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
    return df

def data(datafile):
    file = open(datafile, "r")
    data = []
    for line in file:               # converts the data .txt file to a list of lists 
        file = open(datafile, "r")
        stripped_line = line. strip()
        line_list = stripped_line. split()
        data.append(line_list)
        file.close() 
    
    Date = []
    LST = []
    Seeing = []
    r0 = []
    row = 0
    while row < len(data):             # extract values of Date, Hour (LST) and seeing from data - to new lists each.
        Date.append(data[row][0])
        LST.append(data[row][4])
        Seeing.append(float(data[row][10]))
        r0.append(float(data[row][12]))      # r0 in mm 
        row += 1

    d = {'Date': Date, 'LST': LST, 'Seeing': Seeing, 'r0': r0}
    df = pd.DataFrame(data=d)
    
    time = pd.to_datetime(df['Date'] + ' ' + df['LST'],format = '%d/%m/%Y %H:%M:%S')
    df = pd.DataFrame({'time': time, 'seeing': df.Seeing, 'r0': df.r0 })
    return df 

def splicing (df,start,end):
    """ input: datafile = .txt file of seeing data ("Seeing_Data.txt"), 
    start, end = date and time of first and last measuremnts wanted, in form dd-mm-yy HH:MM:SS'
    site = observation site. no spaces
        output: spliced table of cyclope data, of the night between start_time and end_time"""
    
    for i in list(range(0,len(df))):
        if df.time.iloc[i].hour == 0:            # fix date for hour 00:-- and 01:--
            df.time.iloc[i] += dt.timedelta(days=1)
        elif df.time.iloc[i].hour == 1:
            df.time.iloc[i] += dt.timedelta(days=1)
    mask1 = (df['time'] > start) & (df['time'] <= end) 
    df = df.loc[mask1]
    return df

########## define specific dark image 

darki = fits.open(r'D:\Master\ZWO\2021-04-12_19_27_48Z_dark\dark.fit')
dark = darki[0].data
pscale = 0.604

######## execute function var on each folder in the path ####
for folder in os.listdir(path):
    daf = {'Time':[], 'Date': [], 'Hour': [], 'diffy': [], 'diffx': [] }
    df = pd.DataFrame(daf).set_index('Time')  
    df = pd.concat([df,var(path+'\\'+folder, dark,12,18,pscale)])

######## create a dataframe of the variance of the difference between stars (based on df) #####
vardf = df.resample('1T').var().dropna() # table with variance of distances per minute
vardf.columns = ['var y', 'var x', 'varr', 'varang']

######### DIMM constants and mask parameters #######
d = 0.17 # subapertures distance from hole centers(m)
D = 0.03 # subaperture diameter (m)
zdeg = 59 # angle between star to zenith (deg)
z = (zdeg*np.pi)/180 # angle in rad 
Lambda = 600*(10**-9) # wavelength (m)
b = d/D 

Kl = 0.364*(1 - (0.532*(b**(-1/3))) - 0.024*(b**(-7/3))) # horizontal 
Kt = 0.364*(1 - (0.798*(b**(-1/3))) + 0.018*(b**(-7/3))) # vertical

########## create series of seeing calculated based on the x axis diff (et) and y axis diff (el) : 
# r0l = 0.98*(np.cos(z))**-0.6*Lambda/el 
# seeing at zenith = S0 = S*(1/cos(z))^-0.6

el = ((0.98*((1/np.cos(z))**-0.6))*(D/Lambda)**0.2*((vardf['var x']/Kl)**0.6)).rename('el')
et = ((0.98*(1/np.cos(z))**(-0.6))*(D/Lambda)**0.2*((vardf['var y']/Kl)**0.6)).rename('et')

vardf = vardf.merge(el,on='Time')
vardf = vardf.merge(et, on='Time')
col = vardf.loc[: , "el":"et"]
vardf['seeing zenith'] = col.mean(axis=1)
vardf.index = vardf.index +  timedelta(hours=3)
#vardf = vardf[vardf['seeing zenith'] <= 2* vardf['seeing zenith'].std()]

print("--- %.2f seconds ---" % (time.time()-start_time))

#%%
############ visualize the close-up image, to make sure the centroids locations (by IRAF) are correct
dat = fits.open(r'D:\Master\ZWO\trial\2021-04-11_20_44_39Z\Light_ASIImg_0.02sec_Bin1_19.8C_gain179_2021-04-11_204443_frame0003.fit')[0].data
dat = dat.astype(float)
dat -= dark.astype(float)
dat = dat/np.std(dat)
w = np.where(dat == np.max(dat)) # Brightest pixel position (y,x)
hb = 150 # half box size around w 
yc = int(w[0][0]) # index of centroid row 
xc = int(w[1][0]) # index of centroid column 
window = dat[(yc-hb):(yc+hb), (xc-hb):(xc+hb)] # box limits: data[yc+-hb, xc+-hb]
plt.imshow(window)

########### find centroids of stars using weighted average 
br = np.where(window == np.max(window)) # Brightest pixel position (y,x)
r = 3  # radius of circle area to calculate weighted mean inside 
k = br[0] # estimated center y axis 
h = br[1] # estimated center x axis

xw = []
X = [] 
yw = [] 
Y = []
for y in range(len(window[0])):
    for x in range(len(window[1])):
        if abs((x-h)**2 + (y-k)**2 - r^2) < 18:
            #window[x,y] == 30   # to visualize the location of area that ios calculated 
            xw.append(window[x,y]) 
            yw.append(window[x,y])
            X.append(x)
            Y.append(y)
plt.imshow(window)

wmean= pd.DataFrame({'xpos': X, 'xw': xw, 'ypos': Y, 'yw': yw, })
x_av = (wmean['xpos']*wmean['xw']).sum()/wmean['xw'].sum()  # center of mass position x axis 
y_av = (wmean['ypos']*wmean['yw']).sum()/wmean['yw'].sum()  # center of mass position y axis 

#%%

##### visualize the correlation between DIMM to cyclope results for seeing

datafile = r'D:\Master\fig_sum\Seeing_Data.txt'
start = vardf.index[0] 
end = vardf.index[len(vardf)-1] 
dfcyc = data(datafile)
rel = splicing(dfcyc.copy(),start,end)
rel = rel.set_index("time")
rel.index = rel.index.rename('Time')
rel = rel.resample('1T').mean()
vardf = vardf.merge(rel.seeing, on="Time")

#scatter plots: 
(fig), (ax1,ax2) = plt.subplots(1, 2, sharey = True)
X = mp.dates.date2num(vardf.index)
xformatter = mp.dates.DateFormatter('%H:%M')
ax1.xaxis.set_major_formatter(xformatter)
#ax1.xaxis.set_major_locator(mp.dates.HourLocator(interval = 2))
ax1.scatter(X,vardf['seeing zenith'], c= 'pink', marker = 'o', s = 3)
ax1.set_ylim([0,8])

multi = 5 # multiplication of sigma 
clipsee,low,high = st.sigmaclip(vardf['seeing'],multi+1,multi) 
#ax1.xaxis.set_major_locator(mp.dates.HourLocator(interval = 2))
ax2.scatter(X,clipsee, c= 'olive', marker = 'o', s = 3)

## with seaborn 
import seaborn as sns
tips = sns.load_dataset("tips")
dfsea = pd.melt(vardf[['seeing', 'seeing zenith']], ignore_index = False, var_name= 'cols', value_name = 'vals')
dfsea['Time'] = dfsea.index
g = sns.factorplot(x="Time", y="vals", data=dfsea, hue='cols')
x_dates = dfsea['Time'].dt.strftime('%H:%M').sort_values().unique()
g.set_xticklabels(labels=x_dates)
