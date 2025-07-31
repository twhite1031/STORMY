# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 08:47:58 2023

@author: kbarbs57
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from matplotlib.offsetbox import AnchoredText
import matplotlib.cm as cm
from windrose import WindroseAxes

col_names = ['time','record', 'voltage','ptemperature', 'temperature','rh', 'solar_rad','pressure','wind_speed','wind_dir']

df = pd.read_csv("/data2/white/Downloads/NEBP/lufftdata/CR300Series_2_LufftData_all_data_20231019.dat",header=4,delimiter=",",names=col_names)

wind_speed = df['wind_speed'].values
wind_dir = df['wind_dir'].values
time = df['time'].values
    
# 15:13 UTC first contact
# 16:37 UTC maximum
# 18:10 UTC ends
date_oi = np.where(time == '2023-10-13 16:00:01')
date_oi = int(date_oi[0])
date_fc = np.where(time == '2023-10-14 15:13:00')
date_fc = int(date_fc[0])
date_max = np.where(time == '2023-10-14 16:37:00')
date_max = int(date_max[0])
date_end = np.where(time == '2023-10-14 18:10:00')
date_end = int(date_end[0])

# Plot wind rose
fig = plt.figure(figsize=(14, 8))
ax = WindroseAxes.from_ax()
ax.bar(wind_dir, wind_speed, normed=True, bins=np.arange(0, 5, 0.5))
ax.tick_params(axis='both', which='major', labelsize=20)
for t in ax.get_xticklabels():
     plt.setp(t, fontsize=22, color="k", fontweight="bold")
ax.set_legend(title = 'Wind Speed (m s$^{-1}$)', bbox_to_anchor=(1.05, 1), loc='upper left',borderaxespad=0,prop={'size':'xx-large'})
# l = ax.legend(loc="upper left")
# leg = ax.legend(fontsize = 'x-large')
# leg.set_title("Wind Speed (m s$^{-1}$)", prop = {'size':'xx-large'})
plt.savefig('/data1/white/OUTPUTS/NEBP/lufftplots/annular_wind_rose.png',dpi=300)
plt.show()

# Plot wind direction
fig = plt.figure(figsize=(16, 14))
plt.plot(wind_dir[date_oi:len(wind_dir)],color='k',linewidth=1)
plt.yticks(fontsize=22)
plt.xticks(np.arange(0, len(wind_dir), 3600),['16','18','20','22','00','02','04','06','08','10','12','14','16','18','20','22'],rotation=310,fontsize=22)  

plt.xlabel('Time (UTC)',fontsize=24,fontweight='bold')
plt.ylabel('Wind Direction ($^\circ$)',fontsize=24,fontweight='bold')
plt.axvline(x=date_max-1862,linestyle='--', color='gray')
plt.axvspan(date_fc-1862, date_max-1862, facecolor='peachpuff', alpha=0.5)
plt.axvspan(date_max-1862, date_end-1862, facecolor='gold', alpha=0.5)
plt.title("Wind Direction Beginning at %s UTC" % time[date_oi],fontsize=25,fontweight='bold')
plt.savefig('/data1/white/OUTPUTS/NEBP/lufftplots/20231013_wind_dir.png',dpi=300)
plt.show()

# Plot wind speed
fig = plt.figure(figsize=(16, 14))
plt.plot(wind_speed[date_oi:len(wind_speed)],color='k',linewidth=1)
plt.yticks(fontsize=22)
plt.xticks(np.arange(0, len(wind_dir), 3600),['16','18','20','22','00','02','04','06','08','10','12','14','16','18','20','22'], rotation=310,fontsize=22)  

plt.xlabel('Time (UTC)',fontsize=24,fontweight='bold')
plt.ylabel('Wind Speed (m s$^{-1}$)',fontsize=24,fontweight='bold')
plt.axvline(x=date_max-1862,linestyle='--', color='gray')
plt.axvspan(date_fc-1862, date_max-1862, facecolor='peachpuff', alpha=0.5)
plt.axvspan(date_max-1862, date_end-1862, facecolor='gold', alpha=0.5)
plt.title("Wind Speed Beginning at %s UTC" % time[date_oi],fontsize=25,fontweight='bold')
plt.savefig('/data1/white/OUTPUTS/NEBP/lufftplots/20231013_wind_speed.png',dpi=300)
plt.show()

######################################################
# Average the wind speed and direction every 1-minute
n = 30 # 30 samples per minute
# need to create a list that is divisible by 30
wind_dir_average = np.mean(wind_dir[date_oi:-113].reshape(-1, n), axis=1) 
wind_speed_average = np.mean(wind_speed[date_oi:-113].reshape(-1, n), axis=1)

# These lines were examples of how to determine the times of interest
#test = time[date_oi::30]
#date_fc2 = np.where(test == '2023-10-14 15:13:02')

# Wind direction average 
fig = plt.figure(figsize=(12, 8))
plt.scatter(np.arange(0,1785,1),wind_dir_average,s=10)
plt.yticks(fontsize=22)
plt.xticks(np.arange(0, len(wind_dir_average), 120),['16','18','20','22', '00','02','04', '06','08','10','12','14','16','18','20'], rotation=310,fontsize=22) 
plt.xlabel('Time (UTC)',fontsize=24,fontweight='bold')
plt.ylabel('Wind Direction ($^\circ$)',fontsize=24,fontweight='bold')
plt.axvline(x=1477,linestyle='--', color='gray')
plt.axvspan(1393, 1477, facecolor='peachpuff', alpha=0.5)
plt.axvspan(1477, 1570, facecolor='gold', alpha=0.5)
plt.title("1-Minute Average of Wind Direction Beginning at %s UTC" % time[date_oi],fontsize=25,fontweight='bold')
plt.savefig('/data1/white/OUTPUTS/NEBP/lufftplots/annular_wind_dir_1minave.png',dpi=300)
plt.show()

# Wind speed average
fig = plt.figure(figsize=(12, 8))
plt.scatter(np.arange(0,1785,1),wind_speed_average,s=10)
plt.yticks(fontsize=22)
plt.xticks(np.arange(0, len(wind_speed_average),120),['16','18','20','22','00','02','04', '06','08','10', '12','14','16','18','20'], rotation=310,fontsize=22)  

plt.xlabel('Time (UTC)',fontsize=24,fontweight='bold')
plt.ylabel('Wind Speed (m s$^{-1}$)',fontsize=24,fontweight='bold')
plt.axvline(x=1477,linestyle='--', color='gray')
plt.axvspan(1393, 1477, facecolor='peachpuff', alpha=0.5)
plt.axvspan(1477, 1570, facecolor='gold', alpha=0.5)
plt.title("1-Minute Average of Wind Speed Beginning at %s UTC" % time[date_oi],fontsize=25,fontweight='bold')
plt.savefig('/data1/white/OUTPUTS/NEBP/lufftplots/annular_wind_speed_1minave.png',dpi=300)
plt.show()
######################################################
# Average the wind speed and direction following ASOS guidelines
# 2-minute average of 5-second average --- 24 samples
n = 5 #

