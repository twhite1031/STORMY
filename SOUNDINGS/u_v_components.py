# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 09:33:12 2023

@author: kbarbs57
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This code analyzes the u and v wind components for perturbations.
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

import metpy.calc as mpcalc
from metpy.plots import SkewT
from metpy.units import units

col_names = ['time', 'pressure', 'temperature', 'relative_humidity', 'rh',
             'speed', 'direction', 'geopot', 'dewpoint']

# Column 1: time (UTC)
# Column 2: pressure (hPa)
# Column 3: temperature (C)
# Column 4: relative humidity (%)
# Column 4b: relative humidity
# Column 6: wind speed (m/s)
# Column 7: wind direction (degrees)
# Column 13: dewpoint (C)

df = pd.read_fwf(r"/data2/white/Downloads/NEBP/sondedata" \
                 r"/SUO251700_101423_id1_fix.txt",
                 encoding='latin', header=18,
                 infer_nrows=2000,
                 usecols=[1, 2, 3, 4, 5, 6, 7, 11, 13],
                 names=col_names)

# Drop any rows with all NaN values for T, Td, winds
df = df.dropna(subset=('pressure', 'temperature', 'rh', 'direction',
                       'speed', 'dewpoint', 'geopot'), how='any'
               ).reset_index(drop=True)


df = df.replace('-', np.NaN)

#Will need to think about what to do with the data when the balloon pops 

time = df['time'].values
p = df['pressure'].values*units.hPa
T = df['temperature'].values*units.degC
RH = df['rh'].values*units('%')
wind_speed = df['speed'].values
wind_dir = df['direction'].values
Td = df['dewpoint'].values*units.degC
geopot = df['geopot'].values

wind_speed2 = np.zeros(len(wind_speed))
wind_dir2 = np.zeros(len(wind_dir))
p2 = df['pressure'].values
p3 = np.zeros(len(p))
# Convert variables to floats for mathematical operations
for i in range(0, len(wind_speed)):
    wind_speed2[i] = float(wind_speed[i])*1.944
    wind_dir2[i] = float(wind_dir[i])
    p3[i] = float(p2[i])
    geopot[i] = float(geopot[i])

wind_speed22 = wind_speed2[:]*units("knots")
wind_dir22 = wind_dir2[:]*units.degrees
u, v = mpcalc.wind_components(wind_speed22, wind_dir22)
print(u)
print(v)
# Plot u and v with respect to geopot


fig = plt.figure(figsize=(16, 14))
plt.plot(u,geopot,color='k',linewidth=3,label='u Components')
plt.yticks(fontsize=22)


plt.plot(v,geopot,color='tab:blue',linewidth=3,label='v Components')
plt.yticks(fontsize=22)





plt.xlabel('Wind Speed (m s$^{-1}$)',fontsize=24,fontweight='bold')
plt.ylabel('Geopotential Height',fontsize=24,fontweight='bold')
#plt.axvline(x=date_max-1862,linestyle='--', color='gray')
#plt.axvspan(date_fc-1862, date_max-1862, facecolor='peachpuff', alpha=0.5)
#plt.axvspan(date_max-1862, date_end-1862, facecolor='gold', alpha=0.5)
plt.title("U and V components ",fontsize=25,fontweight='bold')


#plt.savefig('/data1/white/NEBP/lufftplots/20231013_solarrad.png',dpi=300)
plt.show()





