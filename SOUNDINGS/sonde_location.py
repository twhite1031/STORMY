# -*- coding: utf-8 -*-
"""
Spyder Editor

This code plots the path of radiosonde and the height of the sonde. It
also computes the distance. 
"""

import matplotlib.pyplot as plt
import pandas as pd
import os, sys, glob
import numpy as np
import math

import metpy.calc as mpcalc
from metpy.plots import SkewT
from metpy.units import units
from metpy.plots import USCOUNTIES

from scipy.signal import medfilt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

### YOU NEED TO CHANGE THIS DIRECTORY       
indir = "/data1/white/Downloads/NEBP/sondedata/lab9"


filelist = sorted(glob.glob(f'{indir}/SUO*.txt'))
nfiles = len(filelist)

# YOU NEED TO ADD ONE IMPORTANT VARIABLE
col_names = ['time', 'pressure', 'temperature', 'relative_humidity', 'rh',
             'speed', 'direction','lon','lat', 'geopot', 'dewpoint']

# loop through the all of your files in Launches
for i in range(0,nfiles):
# Read the data in
    df = pd.read_fwf(filelist[i],
                 encoding='latin', header=18,
                 infer_nrows=2000,
                 usecols=[1, 2, 3, 4, 5, 6, 7,8, 9, 10, 12],
                 names=col_names)
    
    print("i:", i)
    # YOU WILL NEED TO CHANGE 110:116 based on your directory name
    # You want the date from the text file name, Ex: '101423'
    date = filelist[i][52:58]

    # Replace '-' with np.NaN
    df = df.replace('-', np.NaN)
    
    # Drop any rows with all NaN values for T, Td, winds
    df = df.dropna(subset=('pressure', 'temperature', 'rh', 'direction',
                           'speed', 'lon', 'lat','dewpoint', 'geopot'),
                           how='any'
                   ).reset_index(drop=True)
    
    # Read in the specific variables
    time = df['time'].values
    p = df['pressure'].values*units.hPa
    T = df['temperature'].values*units.degC
    RH = df['rh'].values*units('%')
    wind_speed = df['speed'].values
    wind_dir = df['direction'].values
    Td = df['dewpoint'].values*units.degC
    geopot = df['geopot'].values
    long = df['lon'].values
    lat = df['lat'].values
    
    # Convert from objects to floats
    g3 = np.zeros(len(geopot))
    lat2 = np.zeros(len(lat))
    long2 = np.zeros(len(long))
    
    # Convert variables to floats for mathematical operations
    # YOU NEED TO SAY WHAT THE LOOP COUNTER IS
    for j in range(0, len(wind_speed)):
        g3[j] = float(geopot[j])
        lat2[j] = float(lat[j])
        long2[j] = float(long[j])
    
    # Find times where geopotential decreases 
    # and identify what time it recovers
    decrease_height = np.zeros((len(g3),2))
    for k in range(0,len(g3)-6):
        decrease_height[k,1] = g3[k]
        if g3[k+5]-g3[k] <= 0:
            decrease_height[k,0] = 1
            g3[k] = np.nan
    
    # Figure out where the balloons altitude is greater than it was
    # prior to drop*
    where_recovery = np.where(decrease_height[:] == 1)
    if where_recovery[0].size != 0:
        height_prior_to_loss = g3[where_recovery[0][0]-1]
        if height_prior_to_loss > np.nanmax(g3):
            next_height = np.where(g3 > height_prior_to_loss)

            g4 = g3[0:where_recovery[0][0]-1]
            g5 = g3[next_height[0][0]:]

            # Bring the variables back together as one profile
            geopot = np.concatenate([np.array(g4),np.array(g5)])

        else:
            geopot = g3

    else:
        geopot = g3
 
    # Compute the distance in km by converting lat/long to radians
    # YOU NEED TO ADD THE FUNCTION
    dlon = math.radians(np.nanmax(long2)) - math.radians(np.nanmin(long2))
    dlat = math.radians(lat2[-1]) - math.radians(lat2[0])

    a = math.sin(dlat / 2)**2 + math.cos(lat2[0]) * math.cos(lat2[-1])* math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    R = 6373.0

    distance = R * c
    print(distance)
     
    # Create figure with some maps
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(projection=ccrs.LambertConformal())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.STATES)
    ax.add_feature(cfeature.LAKES)
    ax.add_feature(USCOUNTIES.with_scale('500k'), linewidth=1, edgecolor='black')
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    
    # Add the geometries from the shapefile, pass our styler function
    # What long, lat range are we interested in
    ax.set_extent((-107, -104, 34, 37))
    
    ### NEED TO ENTER A VARIABLE HERE
    plt.scatter(long2[0:len(long2)],lat2[0:len(lat2)],transform=ccrs.PlateCarree(),\
                c=geopot[0:len(geopot)]/1000,cmap='viridis',vmin=0, vmax=36, zorder=10)
    plt.text(-106.85,34.15,"Max Height: %2.2f km" % (np.nanmax(geopot)/1000),\
              transform=ccrs.PlateCarree(),fontsize=18)
   
    ### NEED TO ENTER A VARIABLE HERE
    plt.text(-106.85,34.0,"Distance: %2.2f km" % distance,\
              transform=ccrs.PlateCarree(),fontsize=18)    
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize='22')
    cbar.ax.set_ylim(0, 36)
    cbar.set_label('Geopotential height (km)',fontsize='24')
    ax.set_title("Balloon Launch at %s UTC" %
                    time[0], fontsize=26, fontweight='bold')
    launch_time = time[0].replace(':','')
    plt.savefig('sondelocation_' + launch_time +'_'+ date + '.png',dpi=300)
