#ui-*- coding: utf-8 -*-
"""
Spyder Editor

This code plots Lufft variables.
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from matplotlib.offsetbox import AnchoredText

#Column names of Lufft Variables
col_names = ['time','record', 'voltage','ptemperature', 'temperature','rh', 'solar_rad','pressure','wind_speed','wind_dir']

#Reading in the data file
df = pd.read_csv("/data2/white/Downloads/NEBP/lufftdata/CR300Series_2_LufftData_all_data_20231019.dat",header=4,delimiter=",",names=col_names)

#Assigning variables to each column of data    
temperature = df['temperature'].values
rh = df['rh'].values
solar_rad = df['solar_rad'].values
pressure = df['pressure'].values
wind_speed = df['wind_speed'].values
wind_dir = df['wind_dir'].values
time = df['time'].values
 
#Change variable here for easier use of plotting
obsvar1 = wind_dir
obsvar2 = wind_speed
   
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


#For Averaging Noisy Data
datawindow = 30 #10minutes
average_data = []
for ind in range(len(wind_speed)-datawindow + 1):
	average_data.append(np.mean(wind_speed[ind:ind+datawindow]))

#Averaging any data (mainly wind speed and direction)
average_data2 = []
for ind in range(len(wind_dir-datawindow + 1)):
	average_data2.append(np.mean(wind_dir[ind:ind+datawindow]))
#Create Figure and axes
fig, ax1 = plt.subplots(figsize=(16, 14))

#Axis 1
plt.xticks(np.arange(0, len(wind_speed), 3600),['16','18','20','22','00','02','04','06','08','10', '12','14','16','18','20','22'],rotation=310,fontsize=22)

color='tab:blue'
#Plot data at specific length
ax1.plot(average_data[date_oi:len(wind_speed)],color=color,linewidth=3)

#Set labels
ax1.set_xlabel("Time (UTC)",fontsize=24,fontweight='bold')
ax1.set_ylabel("Wind Speed (m/s)",color=color,fontsize=24,fontweight='bold')
ax1.tick_params(axis='y',labelcolor=color)

#Axis 2
ax2 = ax1.twinx()
color='tab:red'
ax2.set_ylabel('Wind Direction \N{DEGREE SIGN}C',color=color,fontsize=24,fontweight='bold')  # we already handled the x-label with ax1
ax2.plot(average_data2[date_oi:len(wind_dir)],color=color,linewidth=3)
ax2.tick_params(axis='y',labelcolor=color)

#Set an area before maximum annularity, a line for maxiumum annularity, and an area for after maximum annularity
plt.axvline(x=date_max-1862,linestyle='--', color='gray')
plt.axvspan(date_fc-1862, date_max-1862, facecolor='mediumorchid', alpha=0.5)
plt.axvspan(date_max-1862, date_end-1862, facecolor='darkviolet', alpha=0.5)
plt.title("Wind Speed and Wind Direction at %s UTC" % time[date_oi],fontsize=26,fontweight='bold')

fig.tight_layout()
#To save and show the figure
#plt.savefig('/data1/white/OUTPUTS/NEBP/lufftplots/20231013_twinax.png',dpi=300)

plt.show()



