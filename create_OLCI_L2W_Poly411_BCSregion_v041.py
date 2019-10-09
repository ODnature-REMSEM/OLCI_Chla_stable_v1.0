#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: jcardoso
"""

#%% ---------------------------------------------------------------------------
# Import MODULES --------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import RegularGridInterpolator as rgi
from matplotlib.colors import LogNorm
import datetime as dt
import xarray as xr

# Show NetCDF Global Attributes + Variable Information ------------------------
def showAttr(da):
    # See Global Attributes
    globalAttrs = da.attrs
    for attr in globalAttrs:
        print("Global Attr = '" +attr+ "'")
        print("----- Value = '" +str(da.attrs[attr])+ "'")
    # See all the datalayers inside the nc file.
    vars = da.data_vars
    #nvars = len(vars)
    for var_name in vars:
        print("")
        print("-- Variable = '" +str(var_name)+ "'")
        print("----- Shape = " +str(vars[var_name].shape))
        print("-- Var Info = " +str(da.variables[var_name]))
        print("")

#%% Chlorophyll Generation ----------------------------------------------------
# Generate CHL-OC5 (OLCIA_2019_LUT) -------------------------------------------
def OC5_gen(rrs412,rrs443,rrs490,rrs510,rrs560,rows,columns,LUT):
    # Input variables as mentioned in LUT
    xmin = -2
    ymin = -0.2
    xmin560 = 0
    pasx = 0.02
    pasy = 0.0352
    pasx560 = 0.03
    nb = 200

    E412 = 171.4
    E443 = 187.7
    E490 = 192.9
    E510 = 192.7
    E560 = 180.0
    
    nlw_412 = rrs412*E412
    nlw_443 = rrs443*E443
    nlw_490 = rrs490*E490
    nlw_510 = rrs510*E510
    nlw_560 = rrs560*E560
    
    # Determine LUT indices
    Ecor560div510 = E560/E510
    Ecor560div490 = E560/E490
    Ecor560div443 = E560/E443
    
    nlw443div560 = nlw_443/nlw_560
    nlw490div560 = nlw_490/nlw_560
    nlw510div560 = nlw_510/nlw_560
    
    ind412 = (nlw_412 - xmin)/pasx
    ind560 = (nlw_560 - xmin560)/pasx560
    R_oc4 = np.nanmax([(nlw510div560*Ecor560div510),
                       (nlw490div560*Ecor560div490),
                       (nlw443div560*Ecor560div443)], axis=0)
    indil = (R_oc4 - ymin)/pasy
    
    # Create matrix to OC5
    OC5 = np.empty([rows,columns]); OC5.fill(np.nan)

    x = np.array(range(nb))
    y = np.array(range(nb))
    z = np.array(range(nb))
    fn = rgi((x,y,z),LUT)
    
    for row in range(0,rows):
        for column in range(0,columns):
            #Extract CHL value from LUT by interpolation
            if ind412[row,column] >= 0 and ind560[row,column] >= 0 and indil[row,column] >= 0 and ind412[row,column] <= 199 and ind560[row,column] <= 199 and indil[row,column] <= 199:
                # Computing OC5
                OC5[row,column] = fn([ind560[row,column],
                                      ind412[row,column],
                                      indil[row,column]])
        
    return OC5

# Generate CHL-OC4me ---------------------------------------------------------
def OC4_gen(rrs443,rrs490,rrs510,rrs560):
    a0 = 0.4502748
    a1 = -3.259491
    a2 = 3.52271
    a3 = -3.359422
    a4 = 0.949586
    
    allBLUE = np.array([rrs443,rrs490,rrs510])
    BLUEmax = np.max(allBLUE, axis=0)
    R = np.log10(BLUEmax/rrs560)
    
    OC4me = 10**(a0 + (a1*R) + (a2*(R**2)) + (a3*(R**3)) + (a4*(R**4)))
    
    return OC4me

# Generate CHL-Gons -----------------------------------------------------------
def Gons_gen(rhow665,rhow709,rhow779):
    bb = (1.61*rhow779)/(0.082-0.6*rhow779)
    RM = rhow709/rhow665
    # Computing Gons
    Gons = ((RM*(0.70+bb)) - 0.40 - (bb**1.06))/0.016 
    
    return Gons

#%% Chlorophyll Selection -----------------------------------------------------
# Select OC4me ----------------------------------------------------------------
def apply_OC4(OC4me,rhow412,rhow443,rhow490,rhow560,rows,columns):
    rhow412div443 = rhow412/rhow443
    rhow560div490 = rhow560/rhow490
    APPOC4 = np.ones([rows,columns])  ### OC4 OK default
    # High CHL
    APPOC4[np.where(OC4me >= 10)] = 5 # high CHL
    c1 = np.where(APPOC4 == 5, 0, 1)
    # High SPM
    c2 = np.where(np.log10(rhow560) > (-2.588 + 0.676*rhow560div490 - 0.117*rhow560div490*rhow560div490 + 0.4*0.205), 1, 0)
    APPOC4[np.where(c1 + c2 == 2)] = 2  # high SPM
    # High CDOM
    c3 = np.where(rhow412div443 < (1.043 - -0.226*rhow560div490 - 0.056*rhow560div490*rhow560div490 - 0.3*0.212), 1, 0)
    APPOC4[np.where(c1 + c3 == 2)] = 3 # high CDOM
    # high CDOM and high SPM
    APPOC4[np.where(c1 + c2 + c3 == 3)] = 4
    # Low CDOM
    c4 = np.where(rhow412div443 > 1.25, 1, 0)
    APPOC4[np.where(c1 + c4 == 2)] = 6
    
    return APPOC4

# Select OC5 ------------------------------------------------------------------
def apply_OC5(OC5,rhow412,rhow443,rhow490,rhow560,rows,columns):
    rhow412div443 = rhow412/rhow443
    rhow560div490 = rhow560/rhow490
    APPOC5 = np.ones([rows,columns])  ### OC5 OK default
    # High CHL
    APPOC5[np.where(OC5 >= 10)] = 6 # high CHL
    c1 = np.where(APPOC5 == 6, 0, 1)
    # High SPM (a bit)
    c2 = np.where(np.log10(rhow560) > (-2.624 + 0.787*rhow560div490 - 0.125*rhow560div490*rhow560div490  + 1.3*0.239), 1, 0)
    APPOC5[np.where(c1 + c2 == 2)] = 2 # a OC5 "large"
    # High CDOM (a bit)
    c3 = np.where(rhow412div443 < (1.014 - 0.079*rhow560div490 - 0.123*rhow560div490*rhow560div490 - 2.1*0.247), 1, 0)
    APPOC5[np.where(c1 + c3 == 2)] = 2 # a OC5 "large"
    # Very high SPM
    c4 = np.where(np.log10(rhow560) > (-2.624 + 0.787*rhow560div490 - 0.125*rhow560div490*rhow560div490 +2.5*0.239), 1, 0)
    APPOC5[np.where(c1 + c4 == 2)] = 3 # very high SPM
    # Very high CDOM
    c5 = np.where(rhow412div443 < (1.014 - 0.079*rhow560div490 - 0.123*rhow560div490*rhow560div490 - 3.2*0.247), 1, 0)
    APPOC5[np.where(c1 + c5 == 2)] = 4 # very high CDOM
    # high CDOM and high SPM
    APPOC5[np.where(c1 + c4 + c5 == 3)] = 5
    # low CDOM
#    c4 = np.where(rhow412div443  > 1.25, 1, 0)
#    APPOC5[np.where(c1 + c4 == 2)] = 7
    
    return APPOC5

# Select Gons -----------------------------------------------------------------
def apply_Gons(OC4me,rhow665,Gons,rows,columns):
    APPGONS = np.zeros([rows,columns])  # Don't apply GONS algorithm
    # condition 1: OC4me > 8.5
    c1 = np.where(OC4me >= 8.5, 1, 0)
    # condition 2: rhow665 > 0.0081
    c2 = np.where(rhow665 >= 0.0081, 1, 0)
    # condition 3: Gons > 2
    c3 = np.where(Gons > 2, 1, 0)
    APPGONS[np.where(c2 == 0)] = 3 # low SPM
    APPGONS[np.where(c1 + c3 < 2)] = 2 # low CHL
    APPGONS[np.where(c1 + c2 + c3 == 3)] = 1 # Apply Gons

    return APPGONS

# SPM Nechad 2010 -------------------------------------------------------------
def SPM_Nechad2010_gen(rhow560,rhow665,rhow865):
    # Constants
    A560 = 104.66
    C560 = 0.1449
    A665 = 355.85
    C665 = 0.1725
    A865 = 2971.93
    C865 = 0.2115
    
    spm560 = np.array([])
    spm665 = np.array([])
    spm865 = np.array([])
    
    spm560 = (A560*rhow560)/(1-(rhow560/C560))
    spm665 = (A665*rhow665)/(1-(rhow665/C665))
    spm865 = (A865*rhow865)/(1-(rhow865/C865))

    return spm560, spm665, spm865

#%% ---------------------------------------------------------------------------
# Read/Manipulate S3 netCDF dataset -------------------------------------------
olci_l2_folder = '/home/jcardoso/L1_TESTS/olci/l2'
#S3 file
for file_name in sorted(os.listdir(olci_l2_folder)):
    if file_name[-3:]  == '.nc':
        # Get day and time from file (string format)
        year_str = file_name[16:20]
        month_str = file_name[20:22]
        day_str = file_name[22:24]
        hour_str = file_name[25:27]
        minute_str = file_name[27:29]
        second_str = file_name[29:31]
        
        file_path = olci_l2_folder+'/'+file_name
        # Upload netcdf band files
        da = xr.open_dataset(file_path)
        #showAttr(da)

# Check if the file has at least 1 pixel inside the Region of Interesting -----
        lon = da.longitude[:]
        lat = da.latitude[:]

        # Region of Interesting -----------------------------------------------
        lat_north = 52.2
        lat_south = 50.8
        lon_west = 2.0
        lon_east = 3.6

        inds = np.where((lat > lat_south) & (lat < lat_north) & (lon > lon_west) & (lon < lon_east))

        if len(inds[0]) & len(inds[1]) != 0:
            north = np.min(inds[0])
            south = np.max(inds[0])
            west = np.min(inds[1])
            east = np.max(inds[1])
            
### Coordinates
####################
            lon = da.longitude[north:south,west:east]
            lat = da.latitude[north:south,west:east]
            # Create a shape
            rows = np.shape(lat)[0]
            columns = np.shape(lon)[1]

### OLCI mask and SZA
####################
            bitmask = da.bitmask[north:south,west:east]
            sza = da.sza[north:south,west:east]
            
### OLCI bands
####################
            rrs412 = da.Rw412[north:south,west:east]
            rrs443 = da.Rw443[north:south,west:east]
            rrs490 = da.Rw490[north:south,west:east]
            rrs510 = da.Rw510[north:south,west:east]
            rrs560 = da.Rw560[north:south,west:east]
            rrs620 = da.Rw620[north:south,west:east]
            rrs665 = da.Rw665[north:south,west:east]
            rrs709 = da.Rw709[north:south,west:east]
            rrs779 = da.Rw779[north:south,west:east]
            rrs865 = da.Rw865[north:south,west:east]

### POLYMER products
####################
            logchl = da.logchl[north:south,west:east]
            
#%% Apply masks and QC ---------------------------------------------------
            temp = np.where(bitmask == 0, 1, 0)
            temp = np.where(bitmask == 1024, 1, temp)
            temp = np.where(sza <= 75, temp, 0)
            
            rrs412 = np.where(temp == 0, np.nan, rrs412)
            rrs443 = np.where(temp == 0, np.nan, rrs443)
            rrs490 = np.where(temp == 0, np.nan, rrs490)
            rrs510 = np.where(temp == 0, np.nan, rrs510)
            rrs560 = np.where(temp == 0, np.nan, rrs560)
            rrs620 = np.where(temp == 0, np.nan, rrs620)
            rrs665 = np.where(temp == 0, np.nan, rrs665)
            rrs709 = np.where(temp == 0, np.nan, rrs709)
            rrs779 = np.where(temp == 0, np.nan, rrs779)
            rrs865 = np.where(temp == 0, np.nan, rrs865)
            
            logchl = np.where(temp == 0, np.nan, logchl)

#%% Create rhow bands            
            rhow412 = rrs412*np.pi
            rhow443 = rrs443*np.pi
            rhow490 = rrs490*np.pi
            rhow510 = rrs510*np.pi
            rhow560 = rrs560*np.pi
            rhow620 = rrs620*np.pi
            rhow665 = rrs665*np.pi
            rhow709 = rrs709*np.pi
            rhow779 = rrs779*np.pi
            rhow865 = rrs865*np.pi

#%% ---------------------------------------------------------------------------
# DataLayer Chlorophyl (Chla == CHL) ------------------------------------------
# Algorithms Generation
            # OC4me
            OC4me = OC4_gen(rrs443,rrs490,rrs510,rrs560)
            OC4me = np.where(temp == 0, np.nan, OC4me)
            # OC5
            # Upload OLCIA 2019 LUT -------------------------------------------
            # LUT directory
            LUTdir = '/home/jcardoso/Downloads/Matchup_2018/OC5_OLCIA_2019'
            # LUT file
            LUTfile = 'NEW_LUT_olcia_2019_ext_processed.npy'
            LUT = np.load(LUTdir+ "/" +LUTfile)
            OC5 = OC5_gen(rrs412,rrs443,rrs490,rrs510,rrs560,rows,columns,LUT)
            OC5 = np.where(temp == 0, np.nan, OC5)
            # Gons
            Gons = Gons_gen(rhow665,rhow709,rhow779)
            Gons = np.where(temp == 0, np.nan, Gons)
            
# Algorithm Selection
            # OC4me
            APPOC4 = apply_OC4(OC4me,rhow412,rhow443,rhow490,rhow560,rows,columns)
            APPOC4 = np.where(temp == 0, np.nan, APPOC4)
            # OC5
            APPOC5 = apply_OC5(OC5,rhow412,rhow443,rhow490,rhow560,rows,columns)
            APPOC5 = np.where(temp == 0, np.nan, APPOC5)
            # Gons
            APPGONS = apply_Gons(OC4me,rhow665,Gons,rows,columns)
            APPGONS = np.where(temp == 0, np.nan, APPGONS)
            
# Algorithm Combination
            WALGO = np.ones([rows,columns])*999      # No algo apriori
            WALGO[np.where(APPOC4 == 1)] = 1         # OC4/OC5
            c1 = np.where(APPOC5 == 1, 1, 0)
            c2 = np.where(WALGO == 999, 1, 0)
            WALGO[np.where(c1 + c2 == 2)] = 2        # OC5 only
            WALGO[np.where(APPOC5 == 2)] = 3         # OC5 with higher uncertainty
            c1 = np.where(APPOC5 == 1, 1, 0)
            c2 = np.where(APPGONS == 1, 1, 0)
            WALGO[np.where(c1 + c2 == 2)] = 4        # OC5 + Gons
            c1 = np.where(APPGONS == 1, 1, 0)
            c2 = np.where(WALGO == 999, 1, 0)
            c3 = np.where(WALGO == 3, 1, 0)
            WALGO[np.where(c1 + c2 + c3 == 2)] = 5   # only Gons
            WALGO[WALGO == 999] = 6                  # No algo available
            WALGO = np.where(temp == 0, np.nan, WALGO)

# Compute Final Chla
            TTTOC5 = np.empty([rows,columns]); TTTOC5.fill(np.nan)
            TTTOC4 = np.empty([rows,columns]); TTTOC4.fill(np.nan)
            TTTGONS = np.empty([rows,columns]); TTTGONS.fill(np.nan)
            TTTOC5[np.where(APPOC5 == 1)] = OC5[np.where(APPOC5 == 1)]
            TTTOC4[np.where(APPOC4 == 1)] = OC4me[np.where(APPOC4 == 1)]
            TTTGONS[np.where(APPGONS == 1)] = Gons[np.where(APPGONS == 1)]
            CHL3D =  np.array([TTTOC4, TTTOC5, TTTGONS])
            Chla = np.nanmean(CHL3D, axis=0)
            c1 = np.where(np.isnan(Chla), 1, 0)
            c2 = np.where(APPOC5 == 2, 1, 0)
            Chla[np.where(c1 + c2 == 2)] = OC5[np.where(c1 + c2 == 2)]            
            
            
# DataLayer suspended Particulate Matter (SPM) --------------------------------
# Algorithms Generation
            spm560,spm665,spm865 = SPM_Nechad2010_gen(rhow560,rhow665,rhow865)

# Algorithm Selection/Compute
            # Define the RED/NIR ratio to use
            rhow665div865 = np.array([])
            rhow665div865 = rhow665/rhow865
            
            # Fill the map with SPM665 (RED BAND) concentration
            spm = spm665
            bandUse = np.zeros([rows,columns]) # bandUse = 0 - NONE
            bandUse = np.where(spm > 0, 665, bandUse) # bandUse = 665 - SPM560
            
            # RED/NIR < 1.25 - SPM560 (GREEN BAND)
            alpha = np.log(0.002/rhow665)/np.log(0.002/0.001)
            beta = np.log(rhow665/0.001)/np.log(0.002/0.001)
            weight_GREENRED = alpha*spm560 + beta*spm665
            temp = np.array([])
            temp = np.where(rhow665 < 0.002, 1, 0)
            temp = np.where(rhow665 < 0.001, temp+2, temp)
            temp = np.where(rhow560 < 0.050, temp+4, temp)
            temp = np.where(rhow560 < 0.001, temp+8, temp)
            spm = np.where(temp == 5, weight_GREENRED, spm)
            bandUse = np.where(temp == 5, 612, bandUse) # bandUse = 612 - weight_GREENRED
            spm = np.where(temp == 7, spm560, spm)
            bandUse = np.where(temp == 7, 560, bandUse) # bandUse = 560 - SPM560
            spm = np.where(temp == 15, np.nan, spm)
            bandUse = np.where(temp == 15, 0, bandUse) # bandUse = 0 - NONE
            
            # SPM - Use SPM665 (RED BAND) and SPM865 (NIR BAND)
            alpha = np.log(0.09/rhow665)/np.log(0.09/0.05)
            beta = np.log(rhow665/0.05)/np.log(0.09/0.05)
            weight_REDNIR = alpha*spm665 + beta*spm865
            temp = np.array([])
            temp = np.where(rhow665 <= 0.09, 1, 0)
            temp = np.where(rhow665 <= 0.04, temp+2, temp)
            temp = np.where(rhow865 <= 0.05, temp+4, temp)
            temp = np.where(rhow865 <= 0.001, temp+8, temp)
            spm = np.where(temp == 5, weight_REDNIR, spm)
            bandUse = np.where(temp == 5, 765, bandUse) # bandUse = 765 - weight_REDNIR
            spm = np.where(temp == 4, spm865, spm)
            bandUse = np.where(temp == 4, 865, bandUse) # bandUse = 865 - SPM865
            spm = np.where(temp == 0, np.nan, spm) # bandUse should be 1020 but spm1020 doesn't exist
            bandUse = np.where(temp == 0, 1020, bandUse) # bandUse = 1020 - SPM1020
            
            # SPM - Use NIR
            temp = np.array([])
            temp = np.where(rhow665div865 < 1, 1, 0)
            temp = np.where(rhow665 <= 0.05, temp+2, temp)
            temp = np.where(rhow865 <= 0.05, temp+4, temp)
            temp = np.where(rhow865 <= 0.001, temp+8, temp)
            spm = np.where(temp == 1, np.nan, spm) # bandUse should be 1020 but spm1020 doesn't exist
            bandUse = np.where(temp == 1, 1020, bandUse) # bandUse = 1020 - SPM1020
            spm = np.where(temp == 8, spm865, spm)
            bandUse = np.where(temp == 8, 865, bandUse) # bandUse = 865 - SPM865
            
            bandUse = np.where(temp == 0, np.nan, bandUse)
            spm = np.where(temp == 0, np.nan, spm)

#%% Figures and NETCDF generation ---------------------------------------------
# QuickView -------------------------------------------------------------------
            # Create Chla QuickView images
            cmap = plt.get_cmap('rainbow')
            plt.figure(figsize=(6,6))
            plt.imshow(Chla, cmap=cmap, vmin=0.1, vmax=25 , norm=LogNorm())
            cbar = plt.colorbar(orientation="vertical" , ticks=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10, 15, 20])
            cbar.ax.set_yticklabels([0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10, 15, 20])
            cbar.set_label("Chla EUNOSAT (mg m$^{-3}$)")
            plt.xticks(())
            plt.yticks(())
            plt.title("OLCI Chla EUNOSAT | %s/%s/%s - %s:%s:%s" %(day_str,month_str,year_str,hour_str,minute_str,second_str))
            plt.text(columns-170, rows-2, 'v0.4.1 - Poly4.11',{'color':'k'})
            plt.savefig("/home/jcardoso/L1_TESTS/olci/QuickView/OLCI_L2_Poly411_Chla_%s%s%sT%s%s%s_v041.png" %(year_str,month_str,day_str,hour_str,minute_str,second_str))
            plt.close()
            
            cmap = plt.get_cmap('rainbow')
            plt.figure(figsize=(6,6))
            plt.imshow(OC4me, cmap=cmap, vmin=0.1, vmax=25 , norm=LogNorm())
            cbar = plt.colorbar(orientation="vertical" , ticks=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10, 15, 20])
            cbar.ax.set_yticklabels([0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10, 15, 20])
            cbar.set_label("Chla OC4me (mg m$^{-3}$)")
            plt.xticks(())
            plt.yticks(())
            plt.title("OLCI Chla OC4me | %s/%s/%s - %s:%s:%s" %(day_str,month_str,year_str,hour_str,minute_str,second_str))
            plt.text(columns-170, rows-2, 'v0.4.1 - Poly4.11',{'color':'k'})
            plt.savefig("/home/jcardoso/L1_TESTS/olci/QuickView/OLCI_L2_Poly411_OC4me_%s%s%sT%s%s%s_v041.png" %(year_str,month_str,day_str,hour_str,minute_str,second_str))
            plt.close()

            cmap = plt.get_cmap('rainbow')
            plt.figure(figsize=(6,6))
            plt.imshow(OC5, cmap=cmap, vmin=0.1, vmax=25 , norm=LogNorm())
            cbar = plt.colorbar(orientation="vertical" , ticks=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10, 15, 20])
            cbar.ax.set_yticklabels([0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10, 15, 20])
            cbar.set_label("Chla OC5 OLCIA LUT (mg m$^{-3}$)")
            plt.xticks(())
            plt.yticks(())
            plt.title("""OLCI Chla OC5 OLCIA LUT
                           %s/%s/%s - %s:%s:%s""" %(day_str,month_str,year_str,hour_str,minute_str,second_str))
            plt.text(columns-170, rows-2, 'v0.4.1 - Poly4.11',{'color':'k'})
            plt.savefig("/home/jcardoso/L1_TESTS/olci/QuickView/OLCI_L2_Poly411_OC5_OLCIA_LUT_%s%s%sT%s%s%s_v041.png" %(year_str,month_str,day_str,hour_str,minute_str,second_str))
            plt.close()

            # Create Chla QuickView images for Polymer output (name = logchl)
            cmap = plt.get_cmap('rainbow')
            plt.figure(figsize=(6,6))
            plt.imshow(logchl, cmap=cmap, vmin=0.1, vmax=25 , norm=LogNorm())
            cbar = plt.colorbar(orientation="vertical" , ticks=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10, 15, 20])
            cbar.ax.set_yticklabels([0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10, 15, 20])
            cbar.set_label("Chla logchl (mg m$^{-3}$)")
            plt.xticks(())
            plt.yticks(())
            plt.title("OLCI Chla logchl | %s/%s/%s - %s:%s:%s" %(day_str,month_str,year_str,hour_str,minute_str,second_str))
            plt.text(columns-170, rows-2, 'v0.4.1 - Poly4.11',{'color':'k'})
            plt.savefig("/home/jcardoso/L1_TESTS/olci/QuickView/OLCI_L2_Poly411_logchl_%s%s%sT%s%s%s_v041.png" %(year_str,month_str,day_str,hour_str,minute_str,second_str))
            plt.close()

            # Create SPM QuickView images -------------------------------------
            plt.figure(figsize=(6,6))
            plt.imshow(spm, cmap=cmap, vmin=0.1, vmax=70 , norm=LogNorm())
            cbar = plt.colorbar(orientation="vertical" , ticks=[0.1, 1, 2, 5, 15, 30, 45, 65])
            cbar.ax.set_yticklabels([0.1, 1, 2, 5, 15, 30, 45, 65])
            cbar.set_label("SPM Multi-band (g m$^{-3}$)")
            plt.xticks(())
            plt.yticks(())
            plt.title("OLCI SPM Multi-band | %s/%s/%s - %s:%s:%s" %(day_str,month_str,year_str,hour_str,minute_str,second_str))
            plt.text(columns-170, rows-2, 'v0.4.1 - Poly4.11',{'color':'k'})
            plt.savefig("/home/jcardoso/L1_TESTS/olci/QuickView/OLCI_L2_Poly411_SPM_Multi-band_%s%s%sT%s%s%s_v041.png" %(year_str,month_str,day_str,hour_str,minute_str,second_str))
            plt.close()

#%% NetCDF file ---------------------------------------------------------------
            #Directory
            outputdir = '/home/jcardoso/L1_TESTS/olci/l2w/to_ingest'
            #File
            outputfile = "OLCI_L2W_Poly411_Chla_SPM_%s%s%sT%s%s%s_v041.nc" %(year_str,month_str,day_str,hour_str,minute_str,second_str)
            #Path
            outputpath = outputdir+ "/" +outputfile
            
            ### CREATE NETCDF FILE TO WRITE
            # Create Coordinates and Dimensions
            x = columns
            y = rows
            # Create Coordinates and Dimensions
            x_1D = []
            x_min = west
            x_max = east
            x_1D = range(x_min, x_max, 1)
            x_1D = x_1D[::-1]
            
            y_1D = []
            y_min = north
            y_max = south
            y_1D = range(y_min, y_max, 1)
            y_1D = y_1D[::-1]
#            
#            longitude = lon.values
#            latitude = lat.values
#
#            lon = columns
#            lat = rows
            
            # Create Variables
            sentinel3 = xr.Dataset(data_vars={'Chla_ALGO':(('y','x'),WALGO),
                                              'Chla_OC4me':(('y','x'),OC4me),
                                              'Chla_OC5_OLCIA':(('y','x'),OC5),
                                              'Chla_Gons':(('y','x'),Gons),
                                              'Chla_EUNOSAT':(('y','x'),Chla),
                                              'logchl':(('y','x'),logchl),
                                              'SPM_BAND':(('y','x'),bandUse),
                                              'SPM_560':(('y','x'),spm560),
                                              'SPM_665':(('y','x'),spm665),
                                              'SPM_865':(('y','x'),spm865),
                                              'SPM_Multi_Band':(('y','x'),spm),
                                              'Rrs_412':(('y','x'),rrs412),
                                              'Rrs_443':(('y','x'),rrs443),
                                              'Rrs_490':(('y','x'),rrs490),
                                              'Rrs_510':(('y','x'),rrs510),
                                              'Rrs_560':(('y','x'),rrs560),
                                              'Rrs_620':(('y','x'),rrs620),
                                              'Rrs_665':(('y','x'),rrs665),
                                              'Rrs_709':(('y','x'),rrs709),
                                              'Rrs_779':(('y','x'),rrs779),
                                              'Rrs_865':(('y','x'),rrs865),
                                              'longitude':(('y','x'),lon),
                                              'latitude':(('y','x'),lat)},
                                    coords={'y':y_1D, 'x':x_1D, 'lat':(('y','x'),lat), 'lon':(('y','x'),lon)})
            
#%% Attributes
# Global Attributes -----------------------------------------------------------
            sentinel3.attrs['processed_by'] = "Processed by RBINS with Polymer 4.11 AC"
            sentinel3.attrs['title'] = "Sentinel-3A OLCI Rrs, Chla and SPM DataLayers"
            sentinel3.attrs['generated'] = ""
            now = dt.datetime.now()
            sentinel3.attrs['edited'] = now.strftime('%Y-%m-%d %H:%M')
            sentinel3.attrs['sensor'] = "OLCI"
            sentinel3.attrs['region'] = "Belgian Continental Shelf"
            sentinel3.attrs['source'] = "L1 from CODA"
            sentinel3.attrs['platform'] = "Sentinel-3"
            sentinel3.attrs['dname'] = ""
            sentinel3.attrs['contact'] = "jcardoso@naturalsciences.be"
            sentinel3.attrs['origin'] = "L1:CODA, L2:RBINS Poly4.11"
            sentinel3.attrs['project'] = "DCS4COP"
            sentinel3.attrs['time_coverage_start'] = "%s-%s-%sT%s:%s:%s" %(year_str, month_str, day_str, hour_str, minute_str, second_str)
            sentinel3.attrs['time_coverage_end'] = "%s-%s-%sT%s:%s:%s" %(year_str, month_str, day_str, hour_str, minute_str, second_str)
            sentinel3.attrs['geospatial_lat_min'] = str(np.nanmin(lat))
            sentinel3.attrs['geospatial_lat_max'] = str(np.nanmax(lat))
            sentinel3.attrs['geospatial_lon_min'] = str(np.nanmin(lon))
            sentinel3.attrs['geospatial_lon_max'] = str(np.nanmax(lon))
            sentinel3.attrs['raster_width'] = columns
            sentinel3.attrs['raster_height'] = rows
            sentinel3.attrs['raster_resolution'] = "300.0"
            sentinel3.attrs['raster_resolution_unit'] = "meter"
            
# Variable Attributes ---------------------------------------------------------
            sentinel3['latitude'].attrs['units'] = "degrees_north"
            sentinel3['latitude'].attrs['long_name'] = "latitude"
            sentinel3['latitude'].attrs['standard_name'] = "latitude"
            sentinel3['longitude'].attrs['units'] = "degrees_east"
            sentinel3['longitude'].attrs['long_name'] = "longitude"
            sentinel3['longitude'].attrs['standard_name'] = "longitude"
            
            sentinel3['Rrs_412'].attrs['units'] = "sr-1"
            sentinel3['Rrs_412'].attrs['long_name'] = "Rrs 412nm"
            sentinel3['Rrs_412'].attrs['standard_name'] = "surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_radiative_flux_in_air"
            sentinel3['Rrs_412'].attrs['reference'] = ""
            sentinel3['Rrs_412'].attrs['algorithm'] = ""
            sentinel3['Rrs_412'].attrs['ds_flag'] = "Analysis Ready Data"

            sentinel3['Rrs_443'].attrs['units'] = "sr-1"
            sentinel3['Rrs_443'].attrs['long_name'] = "Rrs 443nm"
            sentinel3['Rrs_443'].attrs['standard_name'] = "surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_radiative_flux_in_air"
            sentinel3['Rrs_443'].attrs['reference'] = ""
            sentinel3['Rrs_443'].attrs['algorithm'] = ""
            sentinel3['Rrs_443'].attrs['ds_flag'] = "Analysis Ready Data"

            sentinel3['Rrs_490'].attrs['units'] = "sr-1"
            sentinel3['Rrs_490'].attrs['long_name'] = "Rrs 490nm"
            sentinel3['Rrs_490'].attrs['standard_name'] = "surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_radiative_flux_in_air"
            sentinel3['Rrs_490'].attrs['reference'] = ""
            sentinel3['Rrs_490'].attrs['algorithm'] = ""
            sentinel3['Rrs_490'].attrs['ds_flag'] = "Analysis Ready Data"

            sentinel3['Rrs_510'].attrs['units'] = "sr-1"
            sentinel3['Rrs_510'].attrs['long_name'] = "Rrs 510nm"
            sentinel3['Rrs_510'].attrs['standard_name'] = "surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_radiative_flux_in_air"
            sentinel3['Rrs_510'].attrs['reference'] = ""
            sentinel3['Rrs_510'].attrs['algorithm'] = ""
            sentinel3['Rrs_510'].attrs['ds_flag'] = "Analysis Ready Data"

            sentinel3['Rrs_560'].attrs['units'] = "sr-1"
            sentinel3['Rrs_560'].attrs['long_name'] = "Rrs 560nm"
            sentinel3['Rrs_560'].attrs['standard_name'] = "surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_radiative_flux_in_air"
            sentinel3['Rrs_560'].attrs['reference'] = ""
            sentinel3['Rrs_560'].attrs['algorithm'] = ""
            sentinel3['Rrs_560'].attrs['ds_flag'] = "Analysis Ready Data"

            sentinel3['Rrs_620'].attrs['units'] = "sr-1"
            sentinel3['Rrs_620'].attrs['long_name'] = "Rrs 620nm"
            sentinel3['Rrs_620'].attrs['standard_name'] = "surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_radiative_flux_in_air"
            sentinel3['Rrs_620'].attrs['reference'] = ""
            sentinel3['Rrs_620'].attrs['algorithm'] = ""
            sentinel3['Rrs_620'].attrs['ds_flag'] = "Analysis Ready Data"

            sentinel3['Rrs_665'].attrs['units'] = "sr-1"
            sentinel3['Rrs_665'].attrs['long_name'] = "Rrs 665nm"
            sentinel3['Rrs_665'].attrs['standard_name'] = "surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_radiative_flux_in_air"
            sentinel3['Rrs_665'].attrs['reference'] = ""
            sentinel3['Rrs_665'].attrs['algorithm'] = ""
            sentinel3['Rrs_665'].attrs['ds_flag'] = "Analysis Ready Data"

            sentinel3['Rrs_865'].attrs['units'] = "sr-1"
            sentinel3['Rrs_709'].attrs['long_name'] = "Rrs 709nm"
            sentinel3['Rrs_709'].attrs['standard_name'] = "surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_radiative_flux_in_air"
            sentinel3['Rrs_709'].attrs['reference'] = ""
            sentinel3['Rrs_709'].attrs['algorithm'] = ""
            sentinel3['Rrs_709'].attrs['ds_flag'] = "Analysis Ready Data"

            sentinel3['Rrs_865'].attrs['units'] = "sr-1"
            sentinel3['Rrs_779'].attrs['long_name'] = "Rrs 779nm"
            sentinel3['Rrs_779'].attrs['standard_name'] = "surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_radiative_flux_in_air"
            sentinel3['Rrs_779'].attrs['reference'] = ""
            sentinel3['Rrs_779'].attrs['algorithm'] = ""
            sentinel3['Rrs_779'].attrs['ds_flag'] = "Analysis Ready Data"

            sentinel3['Rrs_865'].attrs['units'] = "sr-1"
            sentinel3['Rrs_865'].attrs['long_name'] = "Rrs 865nm"
            sentinel3['Rrs_865'].attrs['standard_name'] = "surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_radiative_flux_in_air"
            sentinel3['Rrs_865'].attrs['reference'] = ""
            sentinel3['Rrs_865'].attrs['algorithm'] = ""
            sentinel3['Rrs_865'].attrs['ds_flag'] = "Analysis Ready Data"

            sentinel3['Chla_ALGO'].attrs['units'] = "unitless"
            sentinel3['Chla_ALGO'].attrs['long_name'] = "Chla EUNOSAT algorithm selection"
            sentinel3['Chla_ALGO'].attrs['standard_name'] = "Chla_algorithm_selection"
            sentinel3['Chla_ALGO'].attrs['reference'] = "Lavigne et al. (in prep)"
            sentinel3['Chla_ALGO'].attrs['algorithm'] = ""
            sentinel3['Chla_ALGO'].attrs['ds_flag'] = "Analysis Ready Data"

            sentinel3['Chla_OC4me'].attrs['units'] = "mg m-3"
            sentinel3['Chla_OC4me'].attrs['long_name'] = "Chla OC4me"
            sentinel3['Chla_OC4me'].attrs['standard_name'] = "mass_concentration_of_chlorophyll_a_in_sea_water"
            sentinel3['Chla_OC4me'].attrs['reference'] = "Morel and Antoine (2011)"
            sentinel3['Chla_OC4me'].attrs['algorithm'] = ""
            sentinel3['Chla_OC4me'].attrs['ds_flag'] = "Analysis Ready Data"

            sentinel3['Chla_OC5_OLCIA'].attrs['units'] = "mg m-3"
            sentinel3['Chla_OC5_OLCIA'].attrs['long_name'] = "Chla OC5"
            sentinel3['Chla_OC5_OLCIA'].attrs['standard_name'] = "mass_concentration_of_chlorophyll_a_in_sea_water"
            sentinel3['Chla_OC5_OLCIA'].attrs['reference'] = "Gohan et al (2002)"
            sentinel3['Chla_OC5_OLCIA'].attrs['algorithm'] = "New OLCI-A LUT"
            sentinel3['Chla_OC5_OLCIA'].attrs['ds_flag'] = "Analysis Ready Data"

            sentinel3['Chla_Gons'].attrs['units'] = "mg m-3"
            sentinel3['Chla_Gons'].attrs['long_name'] = "Chla Gons"
            sentinel3['Chla_Gons'].attrs['standard_name'] = "mass_concentration_of_chlorophyll_a_in_sea_water"
            sentinel3['Chla_Gons'].attrs['reference'] = "Gons et al (2005)"
            sentinel3['Chla_Gons'].attrs['algorithm'] = ""
            sentinel3['Chla_Gons'].attrs['ds_flag'] = "Analysis Ready Data"

            sentinel3['Chla_EUNOSAT'].attrs['units'] = "mg m-3"
            sentinel3['Chla_EUNOSAT'].attrs['long_name'] = "Chla EUNOSAT S3"
            sentinel3['Chla_EUNOSAT'].attrs['standard_name'] = "mass_concentration_of_chlorophyll_a_in_sea_water"
            sentinel3['Chla_EUNOSAT'].attrs['reference'] = "Lavigne et al. (in prep)"
            sentinel3['Chla_EUNOSAT'].attrs['algorithm'] = ""
            sentinel3['Chla_EUNOSAT'].attrs['ds_flag'] = "Analysis Ready Data"

            sentinel3['logchl'].attrs['units'] = "mg m-3"
            sentinel3['logchl'].attrs['long_name'] = "Chla logchl Polymer output"
            sentinel3['logchl'].attrs['standard_name'] = "mass_concentration_of_chlorophyll_a_in_sea_water"
            sentinel3['logchl'].attrs['reference'] = "François et al (xxxx)"
            sentinel3['logchl'].attrs['algorithm'] = ""
            sentinel3['logchl'].attrs['ds_flag'] = "Analysis Ready Data"

            sentinel3['SPM_BAND'].attrs['units'] = "unitless"
            sentinel3['SPM_BAND'].attrs['long_name'] = "SPM band selection"
            sentinel3['SPM_BAND'].attrs['standard_name'] = "SPM_band_selection"
            sentinel3['SPM_BAND'].attrs['reference'] = "D6.5 - Multi-Sensor Algorithm Improvement Document"
            sentinel3['SPM_BAND'].attrs['algorithm'] = ""
            sentinel3['SPM_BAND'].attrs['ds_flag'] = "Analysis Ready Data"
            
            sentinel3['SPM_560'].attrs['units'] = "g m-3"
            sentinel3['SPM_560'].attrs['long_name'] = "SPM 560nm"
            sentinel3['SPM_560'].attrs['standard_name'] = "mass_concentration_of_suspended_matter_in_sea_water"
            sentinel3['SPM_560'].attrs['reference'] = "Nechad et al. (2010)"
            sentinel3['SPM_560'].attrs['algorithm'] = ""
            sentinel3['SPM_560'].attrs['ds_flag'] = "Analysis Ready Data"

            sentinel3['SPM_665'].attrs['units'] = "g m-3"
            sentinel3['SPM_665'].attrs['long_name'] = "SPM 665nm"
            sentinel3['SPM_665'].attrs['standard_name'] = "mass_concentration_of_suspended_matter_in_sea_water"
            sentinel3['SPM_665'].attrs['reference'] = "Nechad et al. (2010)"
            sentinel3['SPM_665'].attrs['algorithm'] = ""
            sentinel3['SPM_665'].attrs['ds_flag'] = "Analysis Ready Data"
            
            sentinel3['SPM_865'].attrs['units'] = "g m-3"
            sentinel3['SPM_865'].attrs['long_name'] = "SPM 865nm"
            sentinel3['SPM_865'].attrs['standard_name'] = "mass_concentration_of_suspended_matter_in_sea_water"
            sentinel3['SPM_865'].attrs['reference'] = "Nechad et al. (2010)"
            sentinel3['SPM_865'].attrs['algorithm'] = ""
            sentinel3['SPM_865'].attrs['ds_flag'] = "Analysis Ready Data"
            
            sentinel3['SPM_Multi_Band'].attrs['units'] = "g m-3"
            sentinel3['SPM_Multi_Band'].attrs['long_name'] = "SPM Multi_Band"
            sentinel3['SPM_Multi_Band'].attrs['standard_name'] = "mass_concentration_of_suspended_matter_in_sea_water"
            sentinel3['SPM_Multi_Band'].attrs['reference'] = "D6.5 - Multi-Sensor Algorithm Improvement Document"
            sentinel3['SPM_Multi_Band'].attrs['algorithm'] = ""
            sentinel3['SPM_Multi_Band'].attrs['ds_flag'] = "Analysis Ready Data"

            sentinel3.to_netcdf(outputpath, 'w', format='NETCDF4')
            sentinel3.close()

        
        da.close()

# END -------------
