# this code computes the seeing measurements based on measurements done with a DIMM mask.
# it takes all images from a certain path and opens each folder in it to compute the seeing.
# in the end you get a dataframe (vardf) with the calculated variances in x, y, radial and angle, and the seeing at zenith. 

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
import time
import photutils

path = r'D:\Master\ZWO\trial'
os.chdir(path)
start_time = time.time()
##### functions 

def var (path, dark, threshold, fwhm, pscale):  # using IRAF
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
 
        ########## find 2 stars areas       
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

def slice_when(predicate, iterable):
    i, x, size = 0, 0, len(iterable)
    iterable = np.sort(iterable)
    while i < size-1:
        if predicate(iterable[i], iterable[i+1]):
            yield iterable[x:i+1]
            x = i + 1
        i += 1
    yield iterable[x:size]
    
def var2(path, dark, pscale,SNR,fwhm):   # using weighted average 
    """
    input:
    path = path to images folder
    dark = data of dark image
    pscale = plate scale of telescope (arcsec/pixel)
    SNR = signal/noise ratio to use for detecting stars in the close-up window
    
    returns: df = dataframe with stars centroids locations distances on y and x axes, and also angle and radial distance. 
    """
    diffy = []
    diffx = []
    radial = [] 
    angle = [] 
    Date = []
    Hour = []
    Time = [] 
    disqualified = [] 
    for image in os.listdir(path):
        os.chdir(path)  # define the path as current directory
        hdul = fits.open(image)  # Open image
        dat = hdul[0].data  # data = pixel brightness  
        dat = dat.astype(float)
        dat -= dark.astype(float)
        #data -= np.median(data)
        dat = dat/np.std(dat)
        #background = abs(np.median(data))
        
        w = np.where(dat >= 0.95*np.max(dat)) # Brightest pixels position (y,x)
        #w = np.where(data >= 0.9*np.max(data)) 
        hb = 100 # half box size around w 
        yc = int(w[0][0]) # index of centroid row 
        xc = int(w[1][0]) # index of centroid column 
        if yc-hb <0 or yc+hb > len(dat[0]) or xc-hb<0 or xc+hb > len(dat[1]):
            print('The location of the star is less than %i pixels from the edge. \nThis image was discarded: %s' %(hb, image))
            disqualified.append(image)
            continue
        window = dat[(yc-hb):(yc+hb), (xc-hb):(xc+hb)] # box limits: data[yc+-hb, xc+-hb]
        #plt.imshow(window)
        cents = np.where(np.absolute(window) >= SNR) # position of pixels (in window) with values higher than SNR 
        starsy = list(slice_when(lambda x,y: y-x > 10, cents[0].tolist()))
        starsx = list(slice_when(lambda x,y: y-x > 10, cents[1].tolist()))
        starsy = [i for i in starsy if len(i) > 12]
        starsx = [i for i in starsx if len(i) > 12]

        sy = [int(np.mean(starsy[i])) for i in range(len(starsy)) ]
        sx = [int(np.mean(starsx[i])) for i in range(len(starsx)) ]

        if len(sy) < 2 or len (sx) < 2:
            disqualified.append(image)
            print("Only one star found in window - this image was discarded:\n", image)
            continue
        if len(sy) > 2 or len(sx) > 2:
            disqualified.append(image)
            print("more than 2 stars found in window - this image was discarded:\n", image)
            continue
        
        if np.absolute(window[sy[0],sx[0]]) >= SNR:
            br1 = sy[0],sx[0]
            br2 = sy[1],sx[1]
        else:
            br1 = sy[1],sx[0]
            br2 = sy[0],sx[1]
        
        ########### find centroids of stars using weighted average 
        #### centroid of star 1   
        xw = []
        X = [] 
        yw = [] 
        Y = []     
        for y in range(window.shape[0]):
            for x in range(window.shape[1]):
                if abs((x-br1[1])**2 + (y-br1[0])**2 - fwhm^2) < fwhm**1.3:
                    xw.append(window[y,x]) 
                    yw.append(window[y,x])
                    X.append(x)
                    Y.append(y) 
                    """
                    window[y,x] = 30   # to visualize the location of area that ios calculated 
        plt.imshow(window)
        plt.colorbar()   
        """                
        wmean= pd.DataFrame({'xpos': X, 'xw': xw, 'ypos': Y, 'yw': yw })
        x_av1 = (wmean['xpos']*wmean['xw']).sum()/wmean['xw'].sum()  # center of mass position x axis  
        y_av1 = (wmean['ypos']*wmean['yw']).sum()/wmean['yw'].sum()  # center of mass position y axis 
        #x1.append(x_av1)
        #y1.append(y_av1)
        
        #### centroid of star 2 
        xw2 = []
        X2 = [] 
        yw2 = [] 
        Y2 = []
        for y in range(window.shape[0]):
            for x in range(window.shape[1]):
                if abs((x-br2[1])**2 + (y-br2[0])**2 - fwhm^2) < fwhm**1.3:
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
        df.index = df.index + timedelta(hours=3)
        df.drop('Hour',1)
    print('\n\nIn this folder %i images disqualified out of %i for having only 1 / more than 2 stars \n\n' %(len(disqualified), len(os.listdir(path))))    
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
SNR = 5
fwhm = 3

######## execute function var on each folder in the path ####
for folder in os.listdir(path):
    daf = {'Time':[], 'Date': [], 'Hour': [], 'diffy': [], 'diffx': [], 'radial': [], 'angle': [] }
    df = pd.DataFrame(daf).set_index('Time')  
    df = pd.concat([df,var2(path+'\\'+folder,dark,pscale,SNR,fwhm)])
    
nsig = 2
dfn = df.copy()
dfn = dfn[(dfn['diffy'] <= dfn['diffy'].mean() + nsig*dfn['diffy'].std()) & (dfn['diffx'] <= dfn['diffx'].mean() + nsig*dfn['diffx'].std()) ]
dfn = dfn[(dfn['diffy'] >= dfn['diffy'].mean() - nsig*dfn['diffy'].std()) & (dfn['diffx'] >= dfn['diffx'].mean() - nsig*dfn['diffx'].std()) ]
print('%i images disqualified with a %.1f sigma limit' %(len(df)-len(dfn), nsig))

######## create a dataframe of the variance of the difference between stars (based on df) #####
vardf = dfn.resample('1T').var(ddof=0).dropna() # table with variance of distances per minute
vardf.columns = ['var y', 'var x', 'varr', 'varang']

######### DIMM constants and mask parameters #######
d = 0.17 # subapertures distance from hole centers(m)
D = 0.03 # subaperture diameter (m)
zdeg = 59 # angle between star to zenith (deg)
z = (zdeg*np.pi)/180 # angle in rad 
Lambda = 600*(10**-9) # wavelength (m)
b = d/D 

Kl = 0.364*(1 - (0.532*(b**(-1/3))) - 0.024*(b**(-7/3))) # horizontal (x)
Kt = 0.364*(1 - (0.798*(b**(-1/3))) + 0.018*(b**(-7/3))) # vertical (y)

########## create series of seeing calculated based on the x axis diff (et) and y axis diff (el) : 
# r0l = 0.98*(np.cos(z))**-0.6*Lambda/el 
# seeing at zenith = S0 = S*(1/cos(z))^-0.6
#r0l = (vardf['var x']/(Kl*(Lambda**2)*(D**(-1/3))))**(-3/5)
#el1 = 0.98*(Lambda/r0l)*((1/np.cos(z))**(-0.6))*pscale 

el = ((0.98*((1/np.cos(z))**-0.6))*(D/Lambda)**0.2*((vardf['varr']/Kl)**0.6)).rename('el') # y 
et = ((0.98*(1/np.cos(z))**(-0.6))*(D/Lambda)**0.2*((vardf['varang']/Kt)**0.6)).rename('et') # x

vardf = vardf.merge(el,on='Time')
vardf = vardf.merge(et, on='Time')
col = vardf.loc[: , "el":"et"]
vardf['seeing zenith'] = col.mean(axis=1)

#vardf = vardf[vardf['seeing zenith'] <= 2* vardf['seeing zenith'].std()]

print(" ---run time %.2f minutes ---" % (time.time()-start_time))

#%% ########### visualize the close-up image, to make sure the centroids locations (by IRAF) are correct

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
#######

#%% ##### visualize the correlation between DIMM to cyclope results for seeing

datafile = r'D:\Master\fig_sum\Seeing_Data.txt'
start = vardf.index[0] 
end = vardf.index[len(vardf)-1] 
dfcyc = data(datafile)
rel = splicing(dfcyc.copy(),start,end)
rel = rel.set_index("time")
rel.index = rel.index.rename('Time')
rel = rel.resample('1T').mean()
vardf = vardf.merge(rel.seeing, on="Time")

## with seaborn 
import seaborn as sns
tips = sns.load_dataset("tips")
dfsea = pd.melt(vardf[['seeing', 'seeing zenith']], ignore_index = False, var_name= 'cols', value_name = 'vals')
dfsea['Time'] = dfsea.index
g = sns.factorplot(x="Time", y="vals", data=dfsea, hue='cols')
x_dates = dfsea['Time'].dt.strftime('%H:%M').sort_values().unique()
g.set_xticklabels(labels=x_dates, rotation = 30)

#%%
