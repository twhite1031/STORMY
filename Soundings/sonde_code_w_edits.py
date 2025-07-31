import matplotlib.pyplot as plt
import pandas as pd
import os, sys, glob
import numpy as np
import math

import metpy.calc as mpcalc
from metpy.plots import SkewT
from metpy.units import units

from scipy.signal import medfilt

times = []

# Directory where sonde data is stored
indir = "/data2/white/Downloads/NEBP/sondedata/"

filelist = sorted(glob.glob(f'{indir}/SUO25*fix.txt'))
nfiles = len(filelist)
col_names = ['time', 'pressure', 'temperature', 'relative_humidity',
             'speed', 'direction', 'geopot', 'dewpoint']

# Column 1: time (UTC)
# Column 2: pressure (hPa)
# Column 3: temperature (C)
# Column 4: relative humidity (%)
# Column 5: wind speed (m/s)
# Column 6: wind direction (degrees)
# Column 10: geopotential (m)
# Column 12: dewpoint (C)

for i in range(0,16):
# Read the data in
    df = pd.read_csv(filelist[i],
                 encoding='latin', header=18,
                 delim_whitespace=True,
                 usecols=[1, 2, 3, 4, 5, 6, 10, 12],
                 names=col_names)
    
    print("i:", i)
    # YOU WILL NEED TO CHANGE 110:116 based on your directory name
    # You want the date from the text file name, Ex: '101423'
    date = filelist[i][121:127]

    # Replace '-' with np.NaN
    df = df.replace('-', np.NaN)
    
    # Drop any rows with all NaN values for T, Td, winds
    df = df.dropna(subset=('pressure', 'temperature', 'relative_humidity', 
                           'direction','speed', 'dewpoint', 'geopot'), how='any'
                   ).reset_index(drop=True)
    
    # Read in the specific variables
    time = df['time'].values
    p = df['pressure'].values*units.hPa
    T = df['temperature'].values*units.degC
    RH = df['relative_humidity'].values*units('%')
    wind_speed = df['speed'].values
    wind_dir = df['direction'].values
    Td = df['dewpoint'].values*units.degC
    geopot = df['geopot'].values
    
    # Convert from objects to floats
    wind_speed2 = np.zeros(len(wind_speed))
    wind_dir2 = np.zeros(len(wind_dir))
    p2 = df['pressure'].values
    p3 = np.zeros(len(p))
    T2 = df['temperature'].values
    Td2 = df['dewpoint'].values
    T3 = np.zeros(len(T))
    Td3 = np.zeros(len(Td))
    g3 = np.zeros(len(geopot))
    
    # Convert variables to floats for mathematical operations
    for j in range(0, len(wind_speed)):
        wind_speed2[j] = float(wind_speed[j])*1.944
        wind_dir2[j] = float(wind_dir[j])
        p3[j] = float(p2[j])
        T3[j] = float(T2[j])
        Td3[j] = float(Td2[j])
        g3[j] = float(geopot[j])
    
    # Compute the u and v components of wind
    wind_speed22 = wind_speed2[:]*units("knots")
    wind_dir22 = wind_dir2[:]*units.degrees
    u, v = mpcalc.wind_components(wind_speed22, wind_dir22)
    

    # Find times where geopotential decreases 
    # and identify what time it recovers
    decrease_height = np.zeros((len(g3),2))
    for k in range(0,len(g3)-6):
        decrease_height[k,1] = g3[k]
        if g3[k+5]-g3[k] <= 0:
            decrease_height[k,0] = 1
            g3[k] = np.nan
            u[k] = np.nan
            v[k] = np.nan
            T3[k] = np.nan
            Td3[k] = np.nan
            p3[k] = np.nan
            
    print(len(T3),len(Td3),len(p3))
    
    # Figure out where the balloons altitude is greater than it was
    # prior to drop*
    where_recovery = np.where(decrease_height[:] == 1)
    if where_recovery[0].size != 0:
        height_prior_to_loss = g3[where_recovery[0][0]-1]
        if height_prior_to_loss > np.nanmax(g3):
            next_height = np.where(g3 > height_prior_to_loss)
            
            u2 = u[0:where_recovery[0][0]-1]
            u3 = u[next_height[0][0]:]
            v2 = v[0:where_recovery[0][0]-1]
            v3 = v[next_height[0][0]:]
            g4 = g3[0:where_recovery[0][0]-1]
            g5 = g3[next_height[0][0]:]
            T4 = T3[0:where_recovery[0][0]-1]
            T5 = T3[next_height[0][0]:]
            Td4 = Td3[0:where_recovery[0][0]-1]
            Td5 = Td3[next_height[0][0]:]
            p4 = p3[0:where_recovery[0][0]-1]
            p5 = p3[next_height[0][0]:]
        
            # Bring the variables back together as one profile
            u = np.concatenate([np.array(u2),np.array(u3)])
            v = np.concatenate([np.array(v2),np.array(v3)])
            geopot = np.concatenate([np.array(g4),np.array(g5)])
            p = np.concatenate([np.array(p4),np.array(p5)])
            T = np.concatenate([np.array(T4),np.array(T5)])
            Td = np.concatenate([np.array(Td4),np.array(Td5)])
            
        else:
            p = p3
            T = T3
            Td = Td3
            geopot = g3
            u = u
            v = v
    else:
        p = p3
        T = T3
        Td = Td3
        geopot = g3
        v = v
        u = u
       
    print(len(p),len(Td),len(T))   
         
    # Smooth pressure
    smooth_pres = medfilt(p, 7) * units.hPa
    p = np.array(smooth_pres)
    
    # Remove p < 1 mb
    where_lt_1 = np.where(p < 1)
    p[where_lt_1] = np.nan
    
    # Find where the pressure reaches a minimum and only analyze those points
    minmin = np.nanmin(p)     
    minmin = np.where(p == minmin)
    minmin = minmin[0][0]
    p = p[0:minmin,]
    T = T[0:minmin,]
    Td = Td[0:minmin,]
    u = u[0:minmin,]
    v = v[0:minmin,]
    geopot = geopot[0:minmin,]
    
    print(len(p),len(Td),len(T))  
    print(np.nanmin(p))
 
    # Plot the Skew-Ts
    fig = plt.figure(figsize=(9, 9))
    #add_metpy_logo(fig, 115, 100)
    skew = SkewT(fig, rotation=45)
    
    # Plot the data using normal plotting functions, in this case using
    # log scaling in Y, as dictated by the typical meteorological plot.
    skew.plot(p[0:len(p)], T[0:len(T)], 'r')
    skew.plot(p[0:len(p)], Td[0:len(Td)], 'g')
    skew.plot_barbs(p[0::150], u[0::150], v[0::150])
    skew.ax.set_ylim(1000, 5)
    skew.ax.set_xlim(-70, 120)
    skew.ax.set_yticks([1000, 900, 800, 700, 600, 500,
                        400, 300, 200, 100, 50, 25, 10, 5])
    skew.ax.set_xticks([-70,-60, -50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90])
    skew.ax.xaxis.set_tick_params(which='major', labelsize=16, direction='out')
    skew.ax.yaxis.set_tick_params(which='major', labelsize=16, direction='out')
    skew.ax.set_ylabel('Pressure (hPa)', fontsize=24,fontweight='bold')
    skew.ax.set_xlabel('Temperature ($^\circ$C)', fontsize=24,fontweight='bold')
    skew.ax.set_title("Balloon Launch at %s UTC" %
                      time[0], fontsize=26, fontweight='bold')
    
    #props = dict(boxstyle='round', facecolor='white', alpha=1)
    #skew.ax.text(0.05, 0.95, 'Date: %s' % date, transform=skew.ax.transAxes,
    #              fontsize=18, verticalalignment='top',bbox=props)
    
    # Add the relevant special lines
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines()
     
    # Save the plot
    launch_time = time[0].replace(':','')
    #plt.savefig('sonde_' + launch_time +'_'+ date + 'new.png',dpi=300)
    plt.show()
