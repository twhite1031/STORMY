#Code for case study analysis of lufft data (wind_dir)
import datetime
from datetime import time
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from matplotlib.offsetbox import AnchoredText

col_names = ['time','record', 'voltage','ptemperature', 'temperature','rh','solar_rad','pressure','wind_speed','wind_dir']    
filepath = "/data2/white/Downloads/NEBP/lufftdata/CR300Series_2_LufftData_all_data_20231019.dat"
df = pd.read_csv(filepath,header=4,delimiter=",",names=col_names)
    
wind_speed = df['wind_speed'].values
time = df['time'].values

#Important times
#ls=launchstart,sb=startbefore,eb=endbefore,sd=startduring,ed=endduring,
#sa=startafter,ea=endafter,es=eclipsestart,ee=eclipseend,em=eclipsemax

print('Shape of the time variable', time.shape)
date_ls = np.where(time == '2023-10-13 16:00:01')
print('Output of the npwhere of a certain time', date_ls)
print('Index of the npwhere of time: ', time[1862])
date_ls = int(date_ls[0])
print('Output after the npwhere is turned into an int: ',date_ls)
date_sb = np.where(time == '2023-10-14 12:37:00')
date_sb = int(date_sb[0])
date_eb = np.where(time == '2023-10-14 15:37:00')
date_eb = int(date_eb[0])
date_sd = np.where(time == '2023-10-14 15:37:00')
date_sd = int(date_sd[0])
date_ed = np.where(time == '2023-10-14 17:37:00')
date_ed = int(date_ed[0])
date_sa = np.where(time == '2023-10-14 17:37:00')
date_sa = int(date_sa[0])
date_ea = np.where(time == '2023-10-14 20:37:00')
print('Output after endafter is turned into an int', date_ea)
date_ea = int(date_ea[0])
date_es = np.where(time == '2023-10-14 15:13:00')
date_es = int(date_es[0])
date_ee = np.where(time == '2023-10-14 18:10:00')
date_ee = int(date_ee[0])
date_em = np.where(time == '2023-10-14 16:37:00')
date_em = int(date_em[0])
#Find where the eclipse max is with respect to our time frame (before, during, after)
realdate_em = date_em-date_sb
realdate_es = date_es-date_sb
realdate_ee = date_ee-date_sb
print('Time of eclipse max subtracted by our before start', date_em-date_sb)

# Plot exact wind speed
fig = plt.figure(figsize=(15, 12))
plt.plot(wind_speed[date_sb:date_ea],color='k',linewidth=1)
plt.yticks(fontsize=18)

#Had to set arrange to +1 to include the last value of the array (14400), then found the even time interval between 12:37 and 20:37
plt.xticks(np.arange(0, len(wind_speed[date_sb:date_ea])+1, len(wind_speed[date_sb:date_ea])/10),['12:37','13:25','14:13','15:01','15:49','16:37','17:25','18:13','19:01','19:49','20:37'], rotation=310,fontsize=18)  
plt.xlabel('Time (UTC)',fontsize=24,fontweight='bold')
plt.ylabel('Wind Speed (m/s)',fontsize=24,fontweight='bold')

#This line makes the y axis start at the bottom, rather than slightly above the bottom
plt.ylim(ymin=0)
plt.axvline(x=realdate_em,linestyle='--', color='indigo')
plt.axvspan(realdate_es, realdate_em, facecolor='lightsteelblue', alpha=0.5)
plt.axvspan(realdate_em, realdate_ee, facecolor='plum', alpha=0.5)
plt.title("Wind Speed Beginning at %s UTC" % time[date_sb],fontsize=25,fontweight='bold')

#plt.savefig('/data1/white/OUTPUTS/NEBP/casestudy/wind_speed.png',dpi=300)
#plt.show()

#Smooth data
moving_averageseries = df['wind_speed'].rolling(window=30).mean()

#Make the series item into a list (the series plots funny)
moving_average = moving_averageseries.tolist()

#Line plot
fig = plt.figure(figsize=(15, 12))
#data starts way past x=0zero for some reason
plt.plot(moving_average[date_sb:date_ea],color='k',linewidth=1)
plt.yticks(fontsize=22)
plt.xticks(np.arange(0, len(moving_average[date_sb:date_ea])+1, len(moving_average[date_sb:date_ea])/10),['12:37','13:25','14:13','15:01','15:49','16:37','17:25','18:13','19:01','19:49','20:37'],rotation=310,fontsize=22)
plt.xlabel('Time (UTC)',fontsize=24,fontweight='bold')
plt.ylabel('Average Wind Speed (m/s)',fontsize=24,fontweight='bold')
plt.axvline(x=realdate_em,linestyle='--', color='indigo')
plt.axvspan(realdate_es, realdate_em, facecolor='lightsteelblue', alpha=0.5)
plt.axvspan(realdate_em, realdate_ee, facecolor='plum', alpha=0.5)
plt.title("Average Wind Speed Beginning at %s UTC" % time[date_sb],fontsize=25,fontweight='bold')
#plt.savefig('/data1/white/OUTPUTS/NEBP/casestudy/minave_wind_speed.png',dpi=300)
#plt.show()

#Scatter plot
# Plot exact wind speed
fig = plt.figure(figsize=(15, 12))
plt.scatter(np.arange(0,len(wind_speed[date_sb:date_ea]),1),wind_speed[date_sb:date_ea],marker='.')
plt.yticks(fontsize=18)

#Had to set arrange to +1 to include the last value of the array (14400), then found the even time interval between 12:37 and 20:37
plt.xticks(np.arange(0, len(wind_speed[date_sb:date_ea])+1, len(wind_speed[date_sb:date_ea])/10),['12:37','13:25','14:13','15:01','15:49','16:37','17:25','18:13','19:01','19:49','20:37'], rotation=310,fontsize=18)
plt.xlabel('Time (UTC)',fontsize=24,fontweight='bold')
plt.ylabel('Wind Speed (m/s)',fontsize=24,fontweight='bold')

#This line makes the y axis start at the bottom, rather than slightly above the bottom
plt.ylim(ymin=0)
plt.axvline(x=realdate_em,linestyle='--', color='indigo')
plt.axvspan(realdate_es, realdate_em, facecolor='lightsteelblue', alpha=0.5)
plt.axvspan(realdate_em, realdate_ee, facecolor='plum', alpha=0.5)
plt.title("Wind Speed Beginning at %s UTC" % time[date_sb],fontsize=25,fontweight='bold')

#plt.savefig('/data1/white/OUTPUTS/NEBP/casestudy/scatwind_speed.png',dpi=300)
#plt.show()

#Scatter plot
fig = plt.figure(figsize=(15, 12))

#data starts way past x=0 zero for some reason [edit: cause it was a series]
plt.scatter(np.arange(0,len(moving_average[date_sb:date_ea]),1),moving_average[date_sb:date_ea],marker='.')
plt.yticks(fontsize=22)
plt.xticks(np.arange(0, len(moving_average[date_sb:date_ea])+1, len(moving_average[date_sb:date_ea])/10),['12:37','13:25','14:13','15:01','15:49','16:37','17:25','18:13','19:01','19:49','20:37'],rotation=310,fontsize=22)
plt.xlabel('Time (UTC)',fontsize=24,fontweight='bold')
plt.ylabel('Average Wind Speed (m/s)',fontsize=24,fontweight='bold')
plt.axvline(x=realdate_em,linestyle='--', color='indigo')
plt.axvspan(realdate_es, realdate_em, facecolor='lightsteelblue', alpha=0.5)
plt.axvspan(realdate_em, realdate_ee, facecolor='plum', alpha=0.5)
plt.title("Average Wind Speed Beginning at %s UTC" % time[date_sb],fontsize=25,fontweight='bold')
#plt.savefig('/data1/white/OUTPUTS/NEBP/casestudy/scatminave_wind_speed.png',dpi=300)
plt.show()

