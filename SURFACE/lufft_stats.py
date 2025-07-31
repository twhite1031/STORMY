# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 20:01:58 2023

@author: rayne
"""
#Import
import pandas as pd
import numpy as np
import statistics as stats
from tabulate import tabulate

col_names = ['time','record', 'voltage','ptemperature', 'temperature','rh',\
             'solar_rad','pressure','wind_speed','wind_dir']
    
filepath = r"C:\\Users\rayne\OneDrive\Documents\College\Met" \
                 r"\CR300Series_2_LufftData_all_data_20231019.dat"

df = pd.read_csv(filepath,
                 header=4, \
                 delimiter=",",\
                 names=col_names)
    
wind_speed = df['wind_speed'].values
wind_direction = df['wind_dir'].values
time = df['time'].values

#Important times
#sb=startbefore,eb=endbefore,sd=startduring,ed=endduring,
#sa=startafter,ea=endafter,es=eclipsestart,ee=eclipseend,em=eclipsemax
date_sb = np.where(time == '2023-10-14 12:37:00')
date_sb = int(date_sb[0])
date_eb = np.where(time == '2023-10-14 15:37:02')
date_eb = int(date_eb[0])
date_sd = np.where(time == '2023-10-14 15:37:00')
date_sd = int(date_sd[0])
date_ed = np.where(time == '2023-10-14 17:37:02')
date_ed = int(date_ed[0])
date_sa = np.where(time == '2023-10-14 17:37:00')
date_sa = int(date_sa[0])
date_ea = np.where(time == '2023-10-14 20:37:02')
date_ea = int(date_ea[0])
date_es = np.where(time == '2023-10-14 15:13:00')
date_es = int(date_es[0])
date_ee = np.where(time == '2023-10-14 18:10:02')
date_ee = int(date_ee[0])
date_em = np.where(time == '2023-10-14 16:37:00')
date_em = int(date_em[0])

#b=before,d=during,a=after
wind = wind_speed
wind_b = wind[date_sb:date_eb]
wind_d = wind[date_es:date_ee]
wind_a = wind[date_sa:date_ea]

#Calculate statistics
#Mean
mean = np.mean(wind)
mean_b = np.mean(wind_b)
mean_d = np.mean(wind_d)
mean_a = np.mean(wind_a)

#Median
median = np.median(wind)
median_b = np.median(wind_b)
median_d = np.median(wind_d)
median_a = np.median(wind_a)

#Mode
mode = stats.mode(wind)
mode_b = stats.mode(wind_b)
mode_d = stats.mode(wind_d)
mode_a = stats.mode(wind_a)

#Variance
var = stats.variance(wind)
var_b = stats.variance(wind_b)
var_d = stats.variance(wind_d)
var_a = stats.variance(wind_a)

#Standard deviation
stdev = stats.stdev(wind)
stdev_b = stats.stdev(wind_b)
stdev_d = stats.stdev(wind_d)
stdev_a = stats.stdev(wind_a)

#Maximum
max_w = max(wind)
max_b = max(wind_b)
max_d = max(wind_d)
max_a = max(wind_a)

#Minimum
min_w = min(wind)
min_b = min(wind_b)
min_d = min(wind_d)
min_a = min(wind_a)

#Create table
data = [['Mean',mean, mean_b, mean_d, mean_a],\
        ['Median', median, median_b, median_d, median_a],\
        ['Mode', mode, mode_b, mode_d, mode_a],\
        ['Variance', var, var_b, var_d, var_a],\
        ['Standard Deviation', stdev, stdev_b, stdev_d, stdev_a],\
        ['Maximum',max_w,max_b,max_d,max_a],\
        ['Minimum',min_w,min_b,min_d,min_a],]
table = tabulate(data,\
      headers = ['Statistic','Total 30 Hours','2 Hours Before','2 Hours During','2 Hours After'])
print(table)

#Value at eclipse maximum
#m=maximum
wind_m = wind[date_em]
print('Wind at maximum:', wind_m)
