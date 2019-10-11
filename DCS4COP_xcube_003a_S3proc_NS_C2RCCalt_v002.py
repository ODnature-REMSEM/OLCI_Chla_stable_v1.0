#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 20190925
@author: Dimitry Van der Zande
Process S3_L1_EFR files to L2W files using C2RCCalt
input: stardard S3_WFR files (both S3A and S3B) downloaded from CREODIAS, gpt xml file with SNAP protocol
output: L2 S3 tiles with the following DLs: Rrs (21), CHL_NN , CHL_OC4Me, TSM_NN, TSM_Nechad, CHL_EUNOSAT, CHL_OC5, CHL_Gons
*no mosaic added
"""

# %% --------------------------------------------------------------------------
# Import MODULES --------------------------------------------------------------
import os
import shutil
import datetime
import subprocess
import zipfile
import time
import netCDF4 as net4
import numpy as np

import os
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
#from pylab import *
from scipy.interpolate import RegularGridInterpolator as rgi
import warnings
import sys
from netCDF4 import Dataset
import datetime
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl

# %% --------------------------------------------------------------------------
# Definitions    --------------------------------------------------------------
def YMD2str(Y,M,D):
    if M < 10:
        Mstr = '0'
        Mstr = Mstr + str(M)
    else:
        Mstr = str(M)
    if D < 10:
        Dstr = '0'
        Dstr = Dstr + str(D)
    else:
        Dstr = str(D)
    Ystr = str(Y)
    dt = Ystr + Mstr + Dstr
    return dt
def YMDHMnS2str(Y,M,D,H,Mn,S):
    if M < 10:
        Mstr = '0'
        Mstr = Mstr + str(M)
    else:
        Mstr = str(M)
    if D < 10:
        Dstr = '0'
        Dstr = Dstr + str(D)
    else:
        Dstr = str(D)
    if H < 10:
        Hstr = '0'
        Hstr = Hstr + str(H)
    else:
        Hstr = str(H)
    if Mn < 10:
        Mnstr = '0'
        Mnstr = Mnstr + str(Mn)
    else:
        Mnstr = str(Mn)
    if S < 10:
        Sstr = '0'
        Sstr = Sstr + str(S)
    else:
        Sstr = str(S)
    Ystr = str(Y)
    dt = Ystr + Mstr + Dstr + 'T' + Hstr + Mnstr + Sstr
    return dt
def EUNOSATdate2YMD(EUNOSATdate):
    Y = int(str(EUNOSATdate)[0:4])
    M = int(str(EUNOSATdate)[4:6])
    D = int(str(EUNOSATdate)[6:8])

    return Y,M,D
def S3date2YMDHMS(EUNOSATdate):
    Y = int(str(EUNOSATdate)[0:4])
    M = int(str(EUNOSATdate)[4:6])
    D = int(str(EUNOSATdate)[6:8])
    H = int(str(EUNOSATdate)[9:11])
    Mn = int(str(EUNOSATdate)[11:13])
    S = int(str(EUNOSATdate)[13:15])

    return Y,M,D,H,Mn,S
def filelistGenS3_daily(folder,dt,sensor):
    [Ystart,Mstart,Dstart] = EUNOSATdate2YMD(dt)
    dtobj = datetime.date(Ystart,Mstart,Dstart)

    #generate file list for chosen date range
    filelistsensor = []
    filenames_main = next(os.walk(folder))[1]

    str_match = [s for s in filenames_main if sensor in s]
    if len(str_match) >= 1:
        for name in str_match:
            filelistsensor.append(name)
    filelist = []
    str_match = [s for s in filelistsensor if dt in s]
    if len(str_match) >= 1:
        for name in str_match:
            filelist.append(name)
    return filelist
def filelistGenS3_daily_v2(folder,dt,sensor,zipped):
    if zipped == 0:
        [Ystart,Mstart,Dstart] = EUNOSATdate2YMD(dt)
        dtobj = datetime.date(Ystart,Mstart,Dstart)

        #generate file list for datafolder
        filenames_main = next(os.walk(folder))[1]

        #generate sub file list for sensor
        filelistsensor = []
        str_match = [s for s in filenames_main if sensor in s]
        if len(str_match) >= 1:
            for name in str_match:
                filelistsensor.append(name)

        # generate sub file list for sensor and for given day
        filelist = []
        times = getsensingdays(filelistsensor)
        for d in range(0,len(times)):
            if times[d].date() == dtobj:
                filelist.append(filelistsensor[d])
    if zipped == 1:
        [Ystart, Mstart, Dstart] = EUNOSATdate2YMD(dt)
        dtobj = datetime.date(Ystart, Mstart, Dstart)

        # generate file list for datafolder
        filenames_main = next(os.walk(folder))[2]

        # generate sub file list for sensor
        filelistsensor = []
        str_match = [s for s in filenames_main if sensor in s]
        if len(str_match) >= 1:
            for name in str_match:
                filelistsensor.append(name)

        # generate sub file list for sensor and for given day
        filelist = []
        times = getsensingdays(filelistsensor)
        for d in range(0, len(times)):
            if times[d].date() == dtobj:
                filelist.append(filelistsensor[d])
    return filelist

def progressBar(value, endvalue, bar_length=20):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()

def getTiming(filelist_day):
    times = []
    for s in filelist_day:
        t1 = s[16:31]
        t2 = s[32:47]
        [Y1, M1, D1, H1, Mn1, S1] = S3date2YMDHMS(t1)
        [Y2, M2, D2, H2, Mn2, S2] = S3date2YMDHMS(t2)
        dt1 = datetime.datetime(Y1, M1, D1, H1, Mn1, S1)
        dt2 = datetime.datetime(Y2, M2, D2, H2, Mn2, S2)
        times.append(dt1)
        times.append(dt2)
    return times
def getsensingdays(filelist_day):
    times = []
    for s in filelist_day:
        t1 = s[16:31]
        [Y1, M1, D1, H1, Mn1, S1] = S3date2YMDHMS(t1)
        dt1 = datetime.datetime(Y1, M1, D1, H1, Mn1, S1)
        times.append(dt1)
    return times
def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)
def cleanfolder(folderpath):
    for filename in os.listdir(folderpath):
        filepath = os.path.join(folderpath, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)
    return

def fOC5_faster(Rrs412,Rrs443,Rrs490,Rrs510,Rrs560,LUT):
    start_time = time.time()
    print('Generating OC5 DL:')
    iN = np.shape(Rrs412)[0]
    jN = np.shape(Rrs412)[1]
    CHL_OC5 = np.ones([iN, jN]) * -999
    indices = np.transpose(np.where(Rrs412 > 0))
    counter = 1
    [ind412, ind560, indil, fn, loc_valid] = OC5_indices_gen(Rrs412,Rrs443,Rrs490,Rrs510,Rrs560,LUT)

    indices = np.transpose(np.where(loc_valid == 1))
    for pixel in indices:
        progressBar(counter, len(indices), bar_length=20)
        CHL_OC5[pixel[0],pixel[1]] = fn([ind560[pixel[0],pixel[1]], ind412[pixel[0],pixel[1]], indil[pixel[0],pixel[1]]])
        counter = counter + 1

    CHL_OC5[np.where(CHL_OC5 < 0)] = np.nan
    print("Processing time: %s seconds" % (time.time() - start_time))
    return CHL_OC5
def OC5_indices_gen(Rrs412,Rrs443,Rrs490,Rrs510,Rrs560,LUT):
    # prep data for LUT extraction
    # input variables as mentioned in LUT
    xmin = -2
    ymin = -0.2
    xmin560 = 0
    pasx = 0.02
    pasy = 0.0352
    pasx560 = 0.03
    nb = 200
    nlw_412 = Rrs412 * 171.4
    nlw_443 = Rrs443 * 187.7
    nlw_490 = Rrs490 * 192.9
    nlw_510 = Rrs510 * 192.7
    nlw_560 = Rrs560 * 180.0
    E412 = 171.4
    E443 = 187.7
    E490 = 192.9
    E510 = 192.7
    E560 = 180.0
    E665 = 153.1
    # determine LUT indices
    Ecor560div510 = E560 / E510
    Ecor560div490 = E560 / E490
    Ecor560div443 = E560 / E443
    r443div560 = nlw_443 / nlw_560
    r490div560 = nlw_490 / nlw_560
    r510div560 = nlw_510 / nlw_560
    ind412 = (nlw_412 - xmin) / pasx
    ind560 = (nlw_560 - xmin560) / pasx560
    R_oc4_matrix = np.array([r510div560 * Ecor560div510, r490div560 * Ecor560div490, r443div560 * Ecor560div443])
    R_oc4 = np.max(R_oc4_matrix, axis = 0)
    indil = (R_oc4 - ymin) / pasy

    #generate LUT function
    x = np.array(range(nb))
    y = np.array(range(nb))
    z = np.array(range(nb))
    fn = rgi((x, y, z), LUT)

    #generate data pass index
    c1 = np.where(indil > 0, 1, 0)
    c2 = np.where(indil <= 199, 1, 0)
    c3 = np.where(ind412 > 0, 1, 0)
    c4 = np.where(ind412 <= 199, 1, 0)
    c5 = np.where(ind560 > 0, 1, 0)
    c6 = np.where(ind560 <= 199, 1, 0)
    ctotal = c1 + c2 + c3 + c4 + c5 + c6
    loc_valid = np.where(ctotal == 6, 1, 0)
    return ind412, ind560, indil,fn,loc_valid
def OC5_chl_gen_faster(Rrs412,Rrs443,Rrs490,Rrs510,Rrs560,LUT):
#input variables as mentioned in LUT
    xmin = -2
    ymin = -0.2
    xmin560 = 0
    pasx = 0.02
    pasy = 0.0352
    pasx560 = 0.03
    nb = 200
    nlw_412 = Rrs412 * 171.4
    nlw_443 = Rrs443 * 187.7
    nlw_490 = Rrs490 * 192.9
    nlw_510 = Rrs510 * 192.7
    nlw_560 = Rrs560 * 180.0
    E412 = 171.4
    E443 = 187.7
    E490 = 192.9
    E510 = 192.7
    E560 = 180.0
    E665 = 153.1
#determine LUT indices
    Ecor560div510 = E560 / E510
    Ecor560div490 = E560 / E490
    Ecor560div443 = E560 / E443
    r443div560 = nlw_443 / nlw_560
    r490div560 = nlw_490 / nlw_560
    r510div560 = nlw_510 / nlw_560
    ind412 = (nlw_412 - xmin)/pasx
    ind560 = (nlw_560 - xmin560)/pasx560
    R_oc4 = np.max([r510div560 * Ecor560div510,r490div560 * Ecor560div490,r443div560 * Ecor560div443])
    indil = (R_oc4 - ymin)/pasy
#Extract CHL value from LUT by interpolation
    if ind560 > 0 and indil > 0 and ind412 <= 199 and ind560 <= 199 and indil <= 199:
        x = np.array(range(nb))
        y = np.array(range(nb))
        z = np.array(range(nb))
        fn = rgi((x,y,z),LUT)
        chl = fn([ind560,ind412,indil])
    else:
        chl = -999
    return chl


def fOC4(R443, R490, R510, R560):
    allBLUE = np.array([R443, R490, R510])
    BLUEmax = np.max(allBLUE, axis=0)
    R = np.log10(BLUEmax/R560)
    a0 = 0.4502748
    a1 = -3.259491
    a2 = 3.52271
    a3 = -3.359422
    a4 = 0.949586
    CHL = 10**(a0 + a1*R + a2*R**2 + a3*R**3 + a4*R**4)
    return CHL
def fGONS(R665, R709, R779):
    aw1 = 0.4
    aw2 = 0.7
    asp_665 = 0.0146
    p = 1.05
    bb = 1.61*(R779/1)/(0.082-0.6*(R779/1))
    CHL = (1/asp_665)*((R709/R665)*(aw2 + bb) - aw1 - bb**p)
    return CHL
def fSPM_Nechad2010(Rrs560, Rrs665, Rrs865):
    # Constants
    rhow560 = Rrs560 * np.pi
    rhow665 = Rrs665 * np.pi
    rhow865 = Rrs865 * np.pi

    A560 = 104.66
    C560 = 0.1449
    A665 = 355.85
    C665 = 0.1725
    A865 = 2971.93
    C865 = 0.2115

    spm560 = np.array([])
    spm665 = np.array([])
    spm865 = np.array([])

    spm560 = (A560 * rhow560) / (1 - (rhow560 / C560))
    spm665 = (A665 * rhow665) / (1 - (rhow665 / C665))
    spm865 = (A865 * rhow865) / (1 - (rhow865 / C865))

    return spm560, spm665, spm865

def genmap_chl_v6(data, lat, lon, outputname, title, label,region,crange,fpathshape):
    #crange = [0,20]

    # # lat lon coordinates ROI EUNOSAT
    if region == 1:
        lon1 = -8.
        lon2 = 12.
        lat1 = 48.
        lat2 = 60.

    #lat lon coordinates ROI EDULIS
    if region == 2:
        lon1 = 2.
        lon2 = 3.75
        lat1 = 51.
        lat2 = 52.

    # # lat lon coordinates VLIZ Southern North Sea
    if region == 12:
        lon1 = -1
        lon2 = 5
        lat1 = 50
        lat2 = 52.5

    if region == 4:
        lon1 = 2.5
        lon2 = 7
        lat1 = 51.
        lat2 = 56.

    # # lat lon coordinates Kattegat region for Sanyina
    if region == 13:
        lon1 = 10.2
        lon2 = 13
        lat1 = 55.8
        lat2 = 57.6

    # # lat lon coordinates Noordwijk NL region
    if region == 15:
        lon1 = 4.10
        lon2 = 4.75
        lat1 = 52.10
        lat2 = 52.40

    ###set data ranges
    data = np.array(data)

    ###create figure
    fig = plt.figure(figsize=(10, 7.5))
    plt.subplots_adjust(left=0.07, right=0.95, top=0.90, bottom=0.10, wspace=0.15, hspace=0.05)
    ax = plt.subplot(111)

    ###create basemap object
    m = Basemap(projection='cyl', llcrnrlat=lat1, urcrnrlat=lat2, llcrnrlon=lon1, urcrnrlon=lon2,resolution='h')  # res = crude, low, intermediate, high, full
    m.drawcountries(linewidth=0.5)
    m.drawcoastlines(linewidth=0.5)
    #m.drawparallels(np.arange(lat1, lat2, 2.), labels=[1, 0, 0, 0], color='black', labelstyle='+/-',linewidth=0.2,fontsize=14, fontweight='bold')  # draw parallels
    #m.drawmeridians(np.arange(lon1, lon2, 2.), labels=[0, 0, 0, 1], color='black', labelstyle='+/-',linewidth=0.2,fontsize=14, fontweight='bold')  # draw meridians
    m.drawparallels(np.arange(lat1, lat2, 2.), labels=[1, 0, 0, 0], color='black', labelstyle='+/-', linewidth=0.2,
                    fontsize=8, fontweight='bold')  # draw parallels
    m.drawmeridians(np.arange(lon1, lon2, 2.), labels=[0, 0, 0, 1], color='black', labelstyle='+/-', linewidth=0.2,
                    fontsize=8, fontweight='bold')  # draw meridians

    m.fillcontinents()

    ###generate pseudocolor image
    xx, yy = np.meshgrid(lon, lat)
    xx = np.array(xx)
    yy = np.array(yy)

    #img = m.pcolormesh(xx, yy, data, norm=mpl.colors.LogNorm(vmin=crange[0], vmax=crange[1]), cmap='jet')  # log
    img = m.pcolormesh(xx, yy, data, norm=mpl.colors.Normalize(vmin=crange[0], vmax=crange[1]), cmap='jet')   #linear
    cmap = plt.cm.get_cmap("jet")
    cmap.set_under(color='white')
    #cmap.set_over(color='red')


    #colormap
    #cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8])
    #cb = plt.colorbar(ax, cax=cbaxes)
    #cb.set_label(label)

    #cbar = m.colorbar(img, location='bottom', pad="5%")
    cbar = m.colorbar(img, location='bottom', pad=0.4)
    cbar.set_label(label,fontsize=16, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)
    for l in cbar.ax.xaxis.get_ticklabels():
        l.set_weight("bold")
        l.set_fontsize(14)

    #setp(cbar.ax.yaxis.get_ticklabels(), weight='bold', fontsize=14)
    #cbar.ax.tick_params()

    #title
    #plt.title(title,fontsize=18, fontweight='bold')

    #add shape file
    if fpathshape != -1:
        m.readshapefile(fpathshape, 'shape1',linewidth=1)

    #img = m.pcolormesh(xx, yy, data)
    plt.title(title,fontsize=18, fontweight='bold')
    plt.savefig(outputname, dpi=600)
    plt.close()


def apply_OC4_v2(CHL_OC4, R12, R53, R560):
    iN = np.shape(CHL_OC4)[0]
    jN = np.shape(CHL_OC4)[1]
    RES = np.ones([iN, jN])  ### OC4 OK default
## High CHL
    RES[np.where(CHL_OC4 >= 10)] = 5 # high CHL
    c1 = np.where(RES == 5, 0, 1)
### High SPM
    c2 = np.where(np.log10(R560) > -2.592 +0.704*R53 -0.156*R53*R53  +0.08, 1, 0)
    RES[np.where(c1+c2 == 2)] = 2  ## high SPM
### High CDOM
    c3 = np.where(R12 < 1.004 -0.042*R53 -0.067*R53*R53+0.03, 1, 0)
    RES[np.where(c1 + c3 == 2)] = 3 ## high CDOM
### high CDOM and high SPM
    RES[np.where(c1 + c2 + c3 == 3)] = 4 ## high SPM
### low CDOM
    c4 = np.where(R12  > 1.25, 1, 0)
    RES[np.where(c1+c4 == 2)] = 6
    return(RES)
def apply_OC5_v2(CHL_OC5, R12, R53, R560):
    iN = np.shape(CHL_OC5)[0]
    jN = np.shape(CHL_OC5)[1]
    RES = np.ones([iN, jN])  ### OC4 OK default
## High CHL
    RES[np.where(CHL_OC5 >= 10)] = 6 # high CHL
    c1 = np.where(RES == 5, 0, 1)
### High SPM (a bit)
    c2 = np.where(np.log10(R560) > -2.630 +0.803*R53 -0.131*R53*R53  +0.33, 1, 0)
    RES[np.where(c1+c2 == 2)] = 2  ## a OC5 "large"
### High CDOM (a bit)
    c3 = np.where(R12 <  1.012 - 0.070*R53 -0.131*R53*R53-0.54, 1, 0)
    RES[np.where(c1 + c3 == 2)] = 2 ## OC5 "large"
### Very high SPM (a bit)
    c4 = np.where(np.log10(R560) > -2.630 +0.803*R53 -0.131*R53*R53  +0.60, 1, 0)
    RES[np.where(c1+c4 == 2)] = 3  ## very high SPM
### Very high CDOM (a bit)
    c5 = np.where(R12 <  1.012 - 0.070*R53 -0.131*R53*R53-0.80, 1, 0)
    RES[np.where(c1 + c5 == 2)] = 4 ## very high CDOM
### high CDOM and high SPM
    RES[np.where(c1 + c4 + c5 == 3)] = 5 ## high SPM
### low CDOM
    c4 = np.where(R12  > 1.25, 1, 0)
    RES[np.where(c1+c4 == 2)] = 7
    return(RES)
def apply_GONS_v2(CHL_OC4, R665, CHL_GONS):
    iN = np.shape(CHL_OC4)[0]
    jN = np.shape(CHL_OC4)[1]
    RES = np.ones([iN, jN])*0  ### Don't apply GONS algorithm
## condition 1: chl_OC4 > 5
    c1 = np.where(CHL_OC4 >= 8.1, 1, 0)
### condition 2: R665 > 0.002
    c2 = np.where(R665 >= 0.0076, 1, 0)
### condition 4: CHL_GONS > 2
    c3 = np.where(CHL_GONS > 2, 1, 0)
    RES[np.where(c2 == 0)] = 3 ## low SPM
    RES[np.where(c1 + c3 < 2)] = 2 ## low CHL
    RES[np.where(c1 + c2 + c3 == 3)] = 1 ## Apply CHL_GONS algorithm
### high CDOM and high SPM
    return(RES)


def apply_flags_xrarray(band, idepix_flag, c2rcc_flag, SZA):
    band.values[np.where(idepix_flag == 2)] = np.nan
    band.values[np.where(c2rcc_flag == 2)] = np.nan
    band.values[np.where(SZA > 75)] = np.nan
    band.values[np.where(band <= 0)] = np.nan

    return band
def apply_flags_nparray(band, idepix_flag, c2rcc_flag, SZA):
    band[np.where(idepix_flag == 2)] = np.nan
    band[np.where(c2rcc_flag == 2)] = np.nan
    band[np.where(SZA > 75)] = np.nan
    band[np.where(band <= 0)] = np.nan

    return band


# %% -------------------------------------------
# files/folders --------------------------------
#desktop
# S3folder='f:/S3/North Sea/L2standard/2018/S3A/'
# targetfolder = 'f:/S3/North Sea/L2standard_mosaic/'
# processingfolder = 'f:/S3/North Sea/processing/' #this folder is systematically emptied. Be careful!!!!
# dates = [20180101,20181231]
# sensors = ['S3A','S3B']
# zipped = 1
#graphfile = 'd:/Dimitry/OneDrive - Royal Belgian Institute of Natural Sciences/Dimitry/000_Python/BEAMscripts/DCS4COP/mosaic/SNAP_S3_mosaic_WGS84_bandmath_v4_1km.xml'

#laptop
S3folder = "d:/0_data/S3/20190421_experiment/S3/L1/"
targetfolder = 'd:/0_data/S3/20190421_experiment/S3/L2_C2RCC_v5/'
processingfolder = 'd:/0_data/S3/processing/' #this folder is systematically emptied. Be careful!!!!
dates = [20190421,20190422]
sensors = ['S3A','S3B']
zipped = 1
graphfileIDEPIX = 'd:/Dimitry/OneDrive - Royal Belgian Institute of Natural Sciences/Dimitry/000_Python/BEAMscripts/DCS4COP/C2RCCalt/olci_resample_subset_idepix_vicarious_NS_v002.xml'
graphfileC2RCCalt = 'd:/Dimitry/OneDrive - Royal Belgian Institute of Natural Sciences/Dimitry/000_Python/BEAMscripts/DCS4COP/C2RCCalt/olci_resample_subset_C2RCCalt_vicarious_NS_v002.xml'
fpath_vcproperties = 'd:/Dimitry/OneDrive - Royal Belgian Institute of Natural Sciences/Dimitry/000_Python/BEAMscripts/DCS4COP/C2RCCalt/vicarious.properties'


#SNAP files
gptbinfolder = 'c:/Program Files/snap/bin/'

#OC5 files
LUTpath = 'd:/Dimitry/OneDrive - Royal Belgian Institute of Natural Sciences/Dimitry/000_Python/PyCharm/DCS4COP/data/OC5_LUT/OC5_OLCIA_2019/NEW_LUT_olcia_2019_ext_processed.npy'
LUT = np.load(LUTpath)

#mapping options
bmap = 0
quickview = 0

# %% -------------------------------------------
# processor --------------------------------
[Ystart,Mstart,Dstart] = EUNOSATdate2YMD(dates[0])
[Ystop,Mstop,Dstop] = EUNOSATdate2YMD(dates[1])
dtstart = datetime.date(Ystart,Mstart,Dstart)
dtstop = datetime.date(Ystop,Mstop,Dstop)

for single_date in daterange(dtstart, dtstop):
    for sensor in sensors:
        #generate datetime object for day
        dt = YMD2str(single_date.year, single_date.month, single_date.day)

        #generate filelist containing all file names for processed day and sensor
        filelist_day = filelistGenS3_daily_v2(S3folder,dt,sensor,zipped)

        if len(filelist_day) > 0:

            # clear processing folder
            cleanfolder(processingfolder)

            # unzip files to processing folder
            if zipped == 1:
                for file in filelist_day:
                    fpath = S3folder + file
                    print('unzipping file ', file)
                    with zipfile.ZipFile(fpath, 'r') as zip_ref:
                        zip_ref.extractall(processingfolder)

            for file in filelist_day:

                # Get day and time from file (string format)
                year_str = file[11:15]
                month_str = file[15:17]
                day_str = file[17:19]
                hour_str = file[20:22]
                minute_str = file[22:24]
                second_str = file[24:26]


                #Run gpt IDEPIX------------------------------------------
                outputnameIDEPIX = processingfolder + file[:-4] + '_IDEPIX.nc'
                gpt = 'gpt.exe'
                gptPath = os.path.join(gptbinfolder,gpt)
                psubs =[]
                psubs.append(gptPath)
                psubs.append(graphfileIDEPIX)
                psubs.append("-p")
                psubs.append(fpath_vcproperties)
                psubs.append("-f")
                psubs.append("NetCDF4-CF")
                psubs.append("-t")
                psubs.append(outputnameIDEPIX)

                if zipped == 1:
                    #filelist_day_unzipped = filelistGenS3_daily_v2(processingfolder,dt,sensor,0)
                    #psubs.append(processingfolder + filelist_day_unzipped[0])
                    psubs.append(processingfolder + file[:-4] + '.SEN3')
                else:
                    psubs.append(S2folder + file)

                # file to process
                print('IDEPIX Processing: ', file)
                # if zipped == 1:
                #     for fd in filelist_day_unzipped:
                #         print(fd)
                # else:
                #     for fd in filelist_day:
                #         print(fd)
                start_time = time.time()
                process = subprocess.Popen(psubs)
                (output, err) = process.communicate()
                process_status = process.wait()
                print("IDEPIX Processing time: %s seconds" % (time.time() - start_time))

                # Run gpt C2RCCalt------------------------------------------
                outputname_C2RCCalt = processingfolder + file[:-4] + '_C2RCCalt.nc'
                gpt = 'gpt.exe'
                gptPath = os.path.join(gptbinfolder, gpt)
                psubs = []
                psubs.append(gptPath)
                psubs.append(graphfileC2RCCalt)
                psubs.append("-p")
                psubs.append(fpath_vcproperties)
                psubs.append("-f")
                psubs.append("NetCDF4-CF")
                psubs.append("-t")
                psubs.append(outputname_C2RCCalt)

                if zipped == 1:
                    #filelist_day_unzipped = filelistGenS3_daily_v2(processingfolder, dt, sensor, 0)
                    #psubs.append(processingfolder + filelist_day_unzipped[0])
                    psubs.append(processingfolder + file[:-4] + '.SEN3')
                else:
                    psubs.append(S2folder + file)

                # file to process
                print('C2RCC Processing: ', file)
                # if zipped == 1:
                #     for fd in filelist_day_unzipped:
                #         print(fd)
                # else:
                #     for fd in filelist_day:
                #         print(fd)

                start_time = time.time()
                process = subprocess.Popen(psubs)
                (output, err) = process.communicate()
                process_status = process.wait()
                print("C2RCCalt Processing time: %s seconds" % (time.time() - start_time))

                #apply flagging and CHL_EUNOSAT processor ----------------------------------------------
                da_IDEPIX = net4.Dataset(outputnameIDEPIX,'r')  # Dataset is the class behavior to open the file and create an instance of the ncCDF4 class

                # pixel_classif_flags
                idepix = da_IDEPIX.variables['pixel_classif_flags']


                #datalayers
                da_C2RCC = net4.Dataset(outputname_C2RCCalt, 'r')

                quality = da_C2RCC.variables['quality_flags']
                c2rcc = da_C2RCC.variables['c2rcc_flags']
                SZA_values = da_C2RCC.variables['SZA']

                lat = da_C2RCC.variables['lat'][:]
                lon = da_C2RCC.variables['lon'][:]

                # Variable
                rrs412 = np.array(da_C2RCC.variables['rrs_2'])
                rrs443 = np.array(da_C2RCC.variables['rrs_3'])
                rrs490 = np.array(da_C2RCC.variables['rrs_4'])
                rrs510 = np.array(da_C2RCC.variables['rrs_5'])
                rrs560 = np.array(da_C2RCC.variables['rrs_6'])
                rrs620 = np.array(da_C2RCC.variables['rrs_7'])
                rrs665 = np.array(da_C2RCC.variables['rrs_8'])
                rrs709 = np.array(da_C2RCC.variables['rrs_11'])
                rrs779 = np.array(da_C2RCC.variables['rrs_16'])
                rrs865 = np.array(da_C2RCC.variables['rrs_17'])
                chl_nn = np.array(da_C2RCC.variables['conc_chl'])
                spm_nn = np.array(da_C2RCC.variables['conc_tsm'])

                # Clean data
                rrs412[np.where(rrs412 < 0)] = np.nan
                rrs412[np.where(rrs412 > 1)] = np.nan
                rrs443[np.where(rrs443 < 0)] = np.nan
                rrs443[np.where(rrs443 > 1)] = np.nan
                rrs490[np.where(rrs490 < 0)] = np.nan
                rrs490[np.where(rrs490 > 1)] = np.nan
                rrs510[np.where(rrs510 < 0)] = np.nan
                rrs510[np.where(rrs510 > 1)] = np.nan
                rrs560[np.where(rrs560 < 0)] = np.nan
                rrs560[np.where(rrs560 > 1)] = np.nan
                rrs620[np.where(rrs620 < 0)] = np.nan
                rrs620[np.where(rrs620 > 1)] = np.nan
                rrs665[np.where(rrs665 < 0)] = np.nan
                rrs665[np.where(rrs665 > 1)] = np.nan
                rrs709[np.where(rrs709 < 0)] = np.nan
                rrs709[np.where(rrs709 > 1)] = np.nan
                rrs779[np.where(rrs779 < 0)] = np.nan
                rrs779[np.where(rrs779 > 1)] = np.nan
                rrs865[np.where(rrs865 < 0)] = np.nan
                rrs865[np.where(rrs865 > 1)] = np.nan
                chl_nn[np.where(chl_nn <= 0)] = np.nan
                chl_nn[np.where(chl_nn > 200)] = np.nan
                spm_nn[np.where(spm_nn <= 0)] = np.nan
                spm_nn[np.where(spm_nn > 200)] = np.nan

                # %% Apply masks and QC ---------------------------------------------------
                rows = np.shape(lat)[0]
                columns = np.shape(lon)[1]

                ### OLCI mask and SZA
                ####################
                SZA_mean = np.nanmean(SZA_values, axis=1)
                SZA = np.zeros([rows, columns])
                for i in range(0, rows):
                    start = columns / 2
                    SZA_left = SZA_values[i, 0]
                    SZA_right = SZA_values[i, np.shape(SZA)[1] - 1]
                    SZA_steps = np.abs((SZA_left - SZA_right) / len((lon)[1]))
                    for ii in range(0, columns):
                        if SZA_left > SZA_right:
                            SZA[i, ii] = SZA_mean[i] + (start * SZA_steps)
                            start -= 1
                        else:
                            SZA[i, ii] = SZA_mean[i] - (start * SZA_steps)
                            start -= 1

                idepix_flag = np.zeros([rows, columns])
                quality_flag = np.zeros([rows, columns])
                c2rcc_flag = np.zeros([rows, columns])
                idepix = np.array(idepix)
                c2rcc = np.array(c2rcc)


                # Edit Flags ---------------
                # IDEPIX
                idepix_flag_masks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
                idepix_flag_meanings = ["IDEPIX_INVALID", "IDEPIX_CLOUD", "IDEPIX_CLOUD_AMBIGUOUS",
                                        "IDEPIX_CLOUD_SURE", "IDEPIX_CLOUD_BUFFER", "IDEPIX_CLOUD_SHADOW",
                                        "IDEPIX_SNOW_ICE", "IDEPIX_BRIGHT", "IDEPIX_WHITE",
                                        "IDEPIX_COASTLINE", "IDEPIX_LAND"]
                idepix_flag_list = ["IDEPIX_INVALID", "IDEPIX_CLOUD", "IDEPIX_CLOUD_AMBIGUOUS", "IDEPIX_CLOUD_SURE",
                                    "IDEPIX_CLOUD_BUFFER", "IDEPIX_CLOUD_SHADOW", "IDEPIX_SNOW_ICE",
                                    "IDEPIX_BRIGHT", "IDEPIX_COASTLINE", "IDEPIX_LAND"] #flags to be applied

                TMP = [idepix_flag_meanings.index(x) if x in idepix_flag_list else 9999 for x in idepix_flag_meanings]
                INDEX = np.where(np.array(TMP) < 1000)[0]
                for d in INDEX:
                    TEST = idepix_flag_masks[d] & idepix
                    idepix_flag[TEST == idepix_flag_masks[d]] = 2

                # C2RCC QUALITY FLAGS
                c2rcc_flag_masks = [1, 2, 4, 8, 16]
                c2rcc_flag_meanings = ["Rtosa_OOS", "Rtosa_OOR", "Rhow_OOR", "Cloud_risk", "Iop_OOR"]
                c2rcc_flag_list = ["Rtosa_OOR", "Cloud_risk", "Iop_OOR"]

                TMP = [c2rcc_flag_meanings.index(x) if x in c2rcc_flag_list else 9999 for x in c2rcc_flag_meanings]
                INDEX = np.where(np.array(TMP) < 1000)[0]
                for d in INDEX:
                    TEST = c2rcc_flag_masks[d] & c2rcc
                    c2rcc_flag[TEST == c2rcc_flag_masks[d]] = 2

                # Apply Flags ---------------
                rrs412 = apply_flags_nparray(rrs412, idepix_flag, c2rcc_flag, SZA)
                rrs443 = apply_flags_nparray(rrs443, idepix_flag, c2rcc_flag, SZA)
                rrs490 = apply_flags_nparray(rrs490, idepix_flag, c2rcc_flag, SZA)
                rrs510 = apply_flags_nparray(rrs510, idepix_flag, c2rcc_flag, SZA)
                rrs560 = apply_flags_nparray(rrs560, idepix_flag, c2rcc_flag, SZA)
                rrs620 = apply_flags_nparray(rrs620, idepix_flag, c2rcc_flag, SZA)
                rrs665 = apply_flags_nparray(rrs665, idepix_flag, c2rcc_flag, SZA)
                rrs709 = apply_flags_nparray(rrs709, idepix_flag, c2rcc_flag, SZA)
                rrs779 = apply_flags_nparray(rrs779, idepix_flag, c2rcc_flag, SZA)
                rrs865 = apply_flags_nparray(rrs865, idepix_flag, c2rcc_flag, SZA)

                chl_nn = apply_flags_nparray(chl_nn, idepix_flag, c2rcc_flag, SZA)
                spm_nn = apply_flags_nparray(spm_nn, idepix_flag, c2rcc_flag, SZA)

                da_IDEPIX.close()
                da_C2RCC.close()

                iN = np.shape(rrs412)[0]
                jN = np.shape(rrs412)[1]

                #apply flagging -To Do-

                # %% Application algorithms ----------------------------------------------------
                # CHL-OC4
                chl_oc4 = fOC4(rrs443 * np.pi, rrs490 * np.pi, rrs510 * np.pi, rrs560 * np.pi)
                chl_oc4[np.where(np.isnan(rrs443))] = np.nan

                # CHL-OC5
                # chl_oc5 = fOC5(Rrs412, Rrs443, Rrs490, Rrs510, Rrs560, LUT)
                chl_oc5 = fOC5_faster(rrs412, rrs443, rrs490, rrs510, rrs560, LUT)
                chl_oc5[np.where(np.isnan(rrs443))] = np.nan

                # Gons
                chl_gons = fGONS(rrs665 * np.pi, rrs709 * np.pi, rrs779 * np.pi)
                chl_gons[np.where(np.isnan(rrs443))] = np.nan

                # QC for OC4, OC5 and Gons
                APPOC4 = apply_OC4_v2(CHL_OC4=chl_oc4, R12=(rrs412 * np.pi) / (rrs443 * np.pi),
                                      R53=(rrs560 * np.pi) / (rrs490 * np.pi), R560=rrs560 * np.pi)
                APPOC5 = apply_OC5_v2(CHL_OC5=chl_oc5, R12=(rrs412 * np.pi) / (rrs443 * np.pi),
                                      R53=(rrs560 * np.pi) / (rrs490 * np.pi), R560=rrs560 * np.pi)
                APPGONS = apply_GONS_v2(CHL_OC4=chl_oc4, R665=rrs665 * np.pi, CHL_GONS=chl_gons)

                # Algorithm merging ------------------------------------------------
                # 1 = OC4
                # 2 = OC5
                # 3 = OC5ext
                # 4 = OC5 + Gons
                # 5 = Gons
                # 6 = no algo

                WALGO = np.ones([iN, jN]) * 999  ### No algo apriori
                WALGO[np.where(APPOC4 == 1)] = 1  ### OC4/OC5

                c1 = np.where(APPOC5 == 1, 1, 0)
                c2 = np.where(WALGO == 999, 1, 0)
                WALGO[np.where(c1 + c2 == 2)] = 2  ### OC5 only
                WALGO[np.where(APPOC5 == 2)] = 3  ### OC5 with higher uncertainty = OC5ext

                c1 = np.where(APPOC5 == 1, 1, 0)
                c2 = np.where(APPGONS == 1, 1, 0)
                WALGO[np.where(c1 + c2 == 2)] = 4  ### OC5 + GONS

                c1 = np.where(APPGONS == 1, 1, 0)
                c2 = np.where(WALGO == 999, 1, 0)
                c3 = np.where(WALGO == 3, 1, 0)  ###gives priority to OC5ext??
                WALGO[np.where(c1 + c2 + c3 == 2)] = 5  ### only GONS

                WALGO[WALGO == 999] = 7  ### No algo available

                # Compute final Chl-a ---------------------------------------------------------
                TTTOC5 = np.ones([iN, jN]) * np.nan
                TTTOC5ext = np.ones([iN, jN]) * np.nan
                TTTOC4 = np.ones([iN, jN]) * np.nan
                TTTGONS = np.ones([iN, jN]) * np.nan
                TTTNN = np.ones([iN, jN]) * np.nan
                TTTOC5[np.where(APPOC5 == 1)] = chl_oc5[np.where(APPOC5 == 1)]
                TTTOC5ext[np.where(APPOC5 == 2)] = chl_oc5[np.where(APPOC5 == 2)]
                TTTOC4[np.where(APPOC4 == 1)] = chl_oc4[np.where(APPOC4 == 1)]
                TTTGONS[np.where(APPGONS == 1)] = chl_gons[np.where(APPGONS == 1)]
                TTTNN[np.where(chl_nn > 0)] = chl_nn[np.where(chl_nn > 0)]
                chl3D = np.array([TTTOC4, TTTOC5, TTTGONS])
                chl = np.nanmean(chl3D, axis=0)

                # add OC5ext
                c1 = np.where(np.isnan(chl), 1, 0)
                c2 = np.where(APPOC5 == 2, 1, 0)
                chl[np.where(c1 + c2 == 2)] = chl_oc5[np.where(c1 + c2 == 2)]

                # add chl_nn
                chl2D = np.array([chl, TTTNN])
                chlext = np.nanmean(chl2D, axis=0)

                #add SPM_nechad
                [spm560, spm665, spm865] = fSPM_Nechad2010(rrs560, rrs665, rrs865)

                # Save a CHL quick-view image
                if quickview == 1:
                    cmap = plt.get_cmap('rainbow')
                    plt.figure(figsize=(12, 12))
                    plt.imshow(chl, cmap=cmap, vmin=0.1, vmax=25, norm=LogNorm())
                    cbar = plt.colorbar(orientation="vertical", ticks=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10, 15, 20])
                    cbar.ax.set_yticklabels([0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10, 15, 20])
                    cbar.set_label("Chl-a (mg m$^{-3}$)")
                    plt.xticks(())
                    plt.yticks(())
                    plt.title("Merged Chl-a product | %s/%s/%s - %s:%s:%s" % (
                    year_str, month_str, day_str, hour_str, minute_str, second_str))
                    plt.savefig(targetfolder + "/OLCI_L2_CODA_newCHL_NS_%s%s%sT%s%s%s.png" % (
                    year_str, month_str, day_str, hour_str, minute_str, second_str))
                    plt.close()

                # Save a CHL basemap image
                if bmap == 1:
                    outputname = targetfolder + "/OLCI_L2_CODA_newCHL_NS_%s%s%sT%s%s%s_bmap.png" % (
                    year_str, month_str, day_str, hour_str, minute_str, second_str)
                    title = "CHL_NS_%s%s%sT%s%s%s.png" % (
                    year_str, month_str, day_str, hour_str, minute_str, second_str)
                    label = "Chl-a (mg m$^{-3}$)"
                    region = 1
                    crange = [0.1, 25]
                    fpathshape = -1
                    # fpathshape = 'c:/Users/Dimitry/Documents/Dimitry/OneDrive - Royal Belgian Institute of Natural Sciences/Dimitry/000_Python/PyCharm/EUNOSAT/data/HELCOM/OSPAR_modified_test_MultiPolygon'
                    genmap_chl_v6(chl, lat, lon, outputname, title, label, region, crange, fpathshape)

                # save NCDF file ------------------------------------------------
                # Directory
                outputdir = targetfolder
                # File
                outputfile = targetfolder + file[:-4] + '_ARDL.nc'
                # Path
                outputpath = outputfile

                ### CREATE NETCDF FILE TO WRITE
                sentinel3 = Dataset(outputpath, 'w', format='NETCDF4_CLASSIC')

                # Dimensions
                columns = lon.shape[1]
                rows = lat.shape[0]

                sentinel3.createDimension('x', columns)
                sentinel3.createDimension('y', rows)

                # Variables
                VOC4 = sentinel3.createVariable('Chla_OC4', 'f4', ('y', 'x'), zlib=True, fill_value=np.nan)
                VOC5 = sentinel3.createVariable('Chla_OC5', 'f4', ('y', 'x'), zlib=True, fill_value=np.nan)
                VGons = sentinel3.createVariable('Chla_Gons', 'f4', ('y', 'x'), zlib=True, fill_value=np.nan)
                Vchlnn = sentinel3.createVariable('Chla_NN', 'f4', ('y', 'x'), zlib=True, fill_value=np.nan)
                VOC4QC = sentinel3.createVariable('Chla_OC4QC', 'f4', ('y', 'x'), zlib=True, fill_value=np.nan)
                VOC5QC = sentinel3.createVariable('Chla_OC5QC', 'f4', ('y', 'x'), zlib=True, fill_value=np.nan)
                VOC5extQC = sentinel3.createVariable('Chla_OC5extQC', 'f4', ('y', 'x'), zlib=True, fill_value=np.nan)
                VGonsQC = sentinel3.createVariable('Chla_GonsQC', 'f4', ('y', 'x'), zlib=True, fill_value=np.nan)
                VCHL = sentinel3.createVariable('Chla_EUNOSAT', 'f4', ('y', 'x'), zlib=True, fill_value=np.nan)
                VCHLext = sentinel3.createVariable('Chla_EUNOSAT_NN', 'f4', ('y', 'x'), zlib=True, fill_value=np.nan)
                VSPM560 = sentinel3.createVariable('SPM_560', 'f4', ('y', 'x'), zlib=True, fill_value=np.nan)
                VSPM665 = sentinel3.createVariable('SPM_665', 'f4', ('y', 'x'), zlib=True, fill_value=np.nan)
                VSPM865 = sentinel3.createVariable('SPM_865', 'f4', ('y', 'x'), zlib=True, fill_value=np.nan)
                VSPMNN = sentinel3.createVariable('SPM_NN', 'f4', ('y', 'x'), zlib=True, fill_value=np.nan)

                VRrs412 = sentinel3.createVariable('Rrs_412', 'f4', ('y', 'x'), zlib=True, fill_value=np.nan)
                VRrs443 = sentinel3.createVariable('Rrs_443', 'f4', ('y', 'x'), zlib=True, fill_value=np.nan)
                VRrs490 = sentinel3.createVariable('Rrs_490', 'f4', ('y', 'x'), zlib=True, fill_value=np.nan)
                VRrs510 = sentinel3.createVariable('Rrs_510', 'f4', ('y', 'x'), zlib=True, fill_value=np.nan)
                VRrs560 = sentinel3.createVariable('Rrs_560', 'f4', ('y', 'x'), zlib=True, fill_value=np.nan)
                VRrs620 = sentinel3.createVariable('Rrs_620', 'f4', ('y', 'x'), zlib=True, fill_value=np.nan)
                VRrs665 = sentinel3.createVariable('Rrs_665', 'f4', ('y', 'x'), zlib=True, fill_value=np.nan)
                VRrs709 = sentinel3.createVariable('Rrs_709', 'f4', ('y', 'x'), zlib=True, fill_value=np.nan)
                VRrs779 = sentinel3.createVariable('Rrs_779', 'f4', ('y', 'x'), zlib=True, fill_value=np.nan)
                VRrs865 = sentinel3.createVariable('Rrs_865', 'f4', ('y', 'x'), zlib=True, fill_value=np.nan)

                # Vbitmask = sentinel3.createVariable('bitmask', 'int32', ('y', 'x'), fill_value=-9999)
                # Vwqsf = sentinel3.createVariable('WQSF_lsb', 'int32', ('y', 'x'), fill_value=-9999)
                Vwalgo = sentinel3.createVariable('Chla_WALGO', 'f4', ('y', 'x'), zlib=True, fill_value=np.nan)
                VAPPOC4 = sentinel3.createVariable('Chla_APPOC4', 'f4', ('y', 'x'), zlib=True, fill_value=np.nan)
                VAPPOC5 = sentinel3.createVariable('Chla_APPOC5', 'f4', ('y', 'x'), zlib=True, fill_value=np.nan)
                VAPPGONS = sentinel3.createVariable('Chla_APPGONS', 'f4', ('y', 'x'), zlib=True, fill_value=np.nan)

                Vlat = sentinel3.createVariable('lat', 'f4', ('y', 'x'), fill_value=np.nan)
                Vlon = sentinel3.createVariable('lon', 'f4', ('y', 'x'), fill_value=np.nan)

                VOC4[:, :] = chl_oc4[:, :]
                VOC5[:, :] = chl_oc5[:, :]
                VGons[:, :] = chl_gons[:, :]
                Vchlnn[:, :] = chl_nn[:, :]
                VOC4QC[:, :] = TTTOC4[:, :]
                VOC5QC[:, :] = TTTOC5[:, :]
                VOC5extQC[:, :] = TTTOC5ext[:, :]
                VGonsQC[:, :] = TTTGONS[:, :]

                VRrs412[:, :] = rrs412
                VRrs443[:, :] = rrs443
                VRrs490[:, :] = rrs490
                VRrs510[:, :] = rrs510
                VRrs560[:, :] = rrs560
                VRrs620[:, :] = rrs620
                VRrs665[:, :] = rrs665
                VRrs709[:, :] = rrs709
                VRrs779[:, :] = rrs779
                VRrs865[:, :] = rrs865

                VCHL[:, :] = chl[:, :]
                VCHLext[:, :] = chlext[:, :]

                VSPM560[:, :] = spm560[:, :]
                VSPM665[:, :] = spm665[:, :]
                VSPM865[:, :] = spm865[:, :]
                VSPMNN[:, :] = spm_nn[:, :]

                # Vbitmask[:, :] = bitmask[:, :]
                # Vwqsf[:, :] = wqsf[:, :]

                Vwalgo[:, :] = WALGO[:, :]
                VAPPOC4[:, :] = APPOC4[:, :]
                VAPPOC5[:, :] = APPOC5[:, :]
                VAPPGONS[:, :] = APPGONS[:, :]

                Vlat[:,:] = lat[:,:]
                Vlon[:,:] = lon[:,:]

                # %% Attributes
                # Global Attributes -----------------------------------------------------------
                sentinel3.processed_by = "RBINS/OD_Nature/REMSEM with C2RCC alternative AC"
                sentinel3.title = "Sentinel-3 OLCI Rrs, Chla and SPM DataLayers"
                sentinel3.generated = ""
                now = datetime.datetime.now()
                sentinel3.edited = now.strftime('%Y-%m-%d %H:%M')
                sentinel3.sensor = "OLCI"
                sentinel3.region = "Greater North Sea"
                sentinel3.source = "RBINS"
                sentinel3.platform = "Sentinel-3"
                sentinel3.dname = ""
                sentinel3.contact = "jcardoso@naturalsciences.be"
                sentinel3.origin = "CREODIAS"
                sentinel3.project = "DCS4COP"
                sentinel3.time_coverage_start = "%s-%s-%sT%s:%s:%s" % (
                year_str, month_str, day_str, hour_str, minute_str, second_str)
                sentinel3.time_coverage_end = "%s-%s-%sT%s:%s:%s" % (
                year_str, month_str, day_str, hour_str, minute_str, second_str)
                sentinel3.geospatial_lat_min = str(np.nanmin(lat))
                sentinel3.geospatial_lat_max = str(np.nanmax(lat))
                sentinel3.geospatial_lon_min = str(np.nanmin(lon))
                sentinel3.geospatial_lon_man = str(np.nanmax(lon))
                sentinel3.raster_width = str(len(lon))
                sentinel3.raster_height = str(len(lat))
                sentinel3.raster_resolution = "300"
                sentinel3.raster_resolution_unit = "meter"

                # Variable Attributes ---------------------------------------------------------
                # Copy variable attributes
                Vlon.units = "degrees East"
                Vlon.standard_name = "longitude"

                Vlat.units = "degrees North"
                Vlat.standard_name = "latitude"

                VRrs412.units = "sr-1"
                VRrs412.long_name = "Rrs 412nm"
                VRrs412.standard_name = "surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_radiative_flux_in_air"
                VRrs412.reference = "D6.1 - Product Specification Document"
                VRrs412.algorithm = ""
                VRrs412.ds_flag = "Analysis Ready Data"

                VRrs443.units = "sr-1"
                VRrs443.long_name = "Rrs 443nm"
                VRrs443.standard_name = "surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_radiative_flux_in_air"
                VRrs443.reference = "D6.1 - Product Specification Document"
                VRrs443.algorithm = ""
                VRrs443.ds_flag = "Analysis Ready Data"

                VRrs490.units = "sr-1"
                VRrs490.long_name = "Rrs 490nm"
                VRrs490.standard_name = "surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_radiative_flux_in_air"
                VRrs490.reference = "D6.1 - Product Specification Document"
                VRrs490.algorithm = ""
                VRrs490.ds_flag = "Analysis Ready Data"

                VRrs510.units = "sr-1"
                VRrs510.long_name = "Rrs 510nm"
                VRrs510.standard_name = "surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_radiative_flux_in_air"
                VRrs510.reference = "D6.1 - Product Specification Document"
                VRrs510.algorithm = ""
                VRrs510.ds_flag = "Analysis Ready Data"

                VRrs560.units = "sr-1"
                VRrs560.long_name = "Rrs 560nm"
                VRrs560.standard_name = "surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_radiative_flux_in_air"
                VRrs560.reference = "D6.1 - Product Specification Document"
                VRrs560.algorithm = ""
                VRrs560.ds_flag = "Analysis Ready Data"

                VRrs620.units = "sr-1"
                VRrs620.long_name = "Rrs 620nm"
                VRrs620.standard_name = "surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_radiative_flux_in_air"
                VRrs620.reference = "D6.1 - Product Specification Document"
                VRrs620.algorithm = ""
                VRrs620.ds_flag = "Analysis Ready Data"

                VRrs665.units = "sr-1"
                VRrs665.long_name = "Rrs665nm"
                VRrs665.standard_name = "surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_radiative_flux_in_air"
                VRrs665.reference = "D6.1 - Product Specification Document"
                VRrs665.algorithm = ""
                VRrs665.ds_flag = "Analysis Ready Data"

                VRrs709.units = "sr-1"
                VRrs709.long_name = "Rrs709nm"
                VRrs709.standard_name = "surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_radiative_flux_in_air"
                VRrs709.reference = "D6.1 - Product Specification Document"
                VRrs709.algorithm = ""
                VRrs709.ds_flag = "Analysis Ready Data"

                VRrs779.units = "sr-1"
                VRrs779.long_name = "Rrs779nm"
                VRrs779.standard_name = "surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_radiative_flux_in_air"
                VRrs779.reference = "D6.1 - Product Specification Document"
                VRrs779.algorithm = ""
                VRrs779.ds_flag = "Analysis Ready Data"

                VRrs865.units = "sr-1"
                VRrs865.long_name = "Rrs865nm"
                VRrs865.standard_name = "surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_radiative_flux_in_air"
                VRrs865.reference = "D6.1 - Product Specification Document"
                VRrs865.algorithm = ""
                VRrs865.ds_flag = "Analysis Ready Data"

                VOC4.units = "mg m-3"
                VOC4.long_name = "Chla OC4me"
                VOC4.standard_name = "mass_concentration_of_chlorophyll_a_in_sea_water"
                VOC4.algorithm = ""
                VOC4.reference ="Morel and Antoine (2011)"
                VOC4.ds_flag = "Analysis Ready Data"

                VOC5.units = "mg m-3"
                VOC5.long_name = "Chla OC5"
                VOC5.standard_name = "mass_concentration_of_chlorophyll_a_in_sea_water"
                VOC5.algorithm = "new OLCI-A LUT"
                VOC5.reference = "Gohin et al (2002)"
                VOC5.ds_flag = "Analysis Ready Data"

                VGons.units = "mg m-3"
                VGons.long_name = "Chlorophyll concentration."
                VGons.standard_name = "mass_concentration_of_chlorophyll_a_in_sea_water"
                VGons.algorithm = ""
                VGons.reference = "Gons et al (2005)"
                VGons.ds_flag = "Analysis Ready Data"

                VCHL.units = "mg m-3"
                VCHL.long_name = "Chla EUNOSAT S3"
                VCHL.standard_name = "mass_concentration_of_chlorophyll_a_in_sea_water"
                VCHL.algorithm = "D6.1 - Product Specification Document"
                VCHL.reference = "Lavigne et al. (in prep)"
                VCHL.ds_flag = "Analysis Ready Data"

                Vchlnn.units = "mg m-3"
                Vchlnn.long_name = "Chla NN"
                Vchlnn.standard_name = "mass_concentration_of_chlorophyll_a_in_sea_water"
                Vchlnn.algorithm = "C2RCC alternative v2"
                Vchlnn.reference = "Brockmann Consult GmbH"
                Vchlnn.ds_flag = "Analysis Ready Data"

                VSPM560.units = "g m-3"
                VSPM560.long_name = "SPM 560nm"
                VSPM560.standard_name = "mass_concentration_of_suspended_matter_in_sea_water"
                VSPM560.algorithm = ""
                VSPM560.reference = "Nechad et al. (2010)"
                VSPM560.ds_flag = "Analysis Ready Data"

                VSPM665.units = "g m-3"
                VSPM665.long_name = "SPM 665nm"
                VSPM665.standard_name = "mass_concentration_of_suspended_matter_in_sea_water"
                VSPM665.algorithm = ""
                VSPM665.reference = "Nechad et al. (2010)"
                VSPM665.ds_flag = "Analysis Ready Data"

                VSPM865.units = "g m-3"
                VSPM865.long_name = "SPM 865nm"
                VSPM865.standard_name = "mass_concentration_of_suspended_matter_in_sea_water"
                VSPM865.algorithm = ""
                VSPM865.reference = "Nechad et al. (2010)"
                VSPM865.ds_flag = "Analysis Ready Data"

                VSPMNN.units = "g m-3"
                VSPMNN.long_name = "SPM NN"
                VSPMNN.standard_name = "mass_concentration_of_suspended_matter_in_sea_water"
                VSPMNN.algorithm = "C2RCC alternative v2"
                VSPMNN.reference = "Brockmann Consult GmbH"
                VSPMNN.ds_flag = "Analysis Ready Data"


                Vwalgo.long_name = "Code to apply the different Chl-a algorithms."
                Vwalgo.ds_flag = "OC4_OC5: 1, OC5_only: 2, OC5_with_high_uncertainty: 3, OC5+Gons_average: 4, Gons: 5, No_algorithm: 6"

                # Vwqsf.long_name = "Classification flags, quality and science flags for Marine and Inland Waters pixels"
                # Vwqsf.flag_masks = [1, 2, 4, 8, 8388608, 16777216, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
                #                   65536, 131072, 262144, 524288, 2097152, 33554432, 67108864, 134217728, 268435456]
                # Vwqsf.flag_meanings = "INVALID WATER LAND CLOUD CLOUD_AMBIGUOUS CLOUD_MARGIN SNOW_ICE INLAND_WATER TIDAL COSMETIC SUSPECT HISOLZEN SATURATED MEGLINT HIGHGLINT WHITECAPS ADJAC WV_FAIL PAR_FAIL AC_FAIL OC4ME_FAIL OCNN_FAIL KDM_FAIL BPAC_ON WHITE_SCATT LOWRW HIGHRW"

                # Vbitmask.long_name = "Polymer flags."
                # Vbitmask.flag_masks = [1, 2, 4, 8, 16, 32, 64, 128, 512, 1024, 2048]
                # Vbitmask.flag_meanings = "LAND CLOUD_BASE L1_INVALID NEGATIVE_BB OUT_OF_BOUNDS EXCEPTION THICK_AEROSOL HIGH_AIR_MASS EXTERNAL_MASK CASE2 INCONSISTENCY"

                sentinel3.close()

            # END -------------


            # clear processing folder
            cleanfolder(processingfolder)







# END -------------