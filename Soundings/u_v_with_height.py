# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 13:38:30 2023

@author: kbarbs57
"""

import matplotlib.pyplot as plt
import pandas as pd
import os, sys, glob
import numpy as np

import metpy.calc as mpcalc
from metpy.cbook import get_test_data
from metpy.plots import add_metpy_logo, SkewT
from metpy.units import units

from scipy.signal import medfilt
from sklearn.linear_model import LinearRegression

# YOU NEED TO CHANGE THE PATH
indir = "/data2/white/Downloads/NEBP/sondedata"

filelist = sorted(glob.glob(f'{indir}/SUO25*fix.txt'))
nfiles = len(filelist)
print('nfiles: ', nfiles)
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

# Loop through the files in 
for i in range(0,nfiles):
    if i == 0:
        # Read the data in
        df = pd.read_fwf(filelist[i],
                     encoding='latin', header=18,
                     infer_nrows=2000,
                     usecols=[1, 2, 3, 4, 5, 6, 7, 11, 13],
                     names=col_names)
        
        # Replace '-' with np.NaN
        df = df.replace('-', np.NaN)
        
        # Drop any rows with all NaN values for T, Td, winds
        df = df.dropna(subset=('pressure', 'temperature', 'rh', 'direction',
                               'speed', 'dewpoint', 'geopot'), how='any'
                       ).reset_index(drop=True)
        
        date = filelist[i][115:121]
        
        time = df['time'].values     #1
        p = df['pressure'].values*units.hPa    #2
        T = df['temperature'].values* units.degC  #3
        RH = df['rh'].values*units('%')  #4
        wind_speed = df['speed'].values  #5
        wind_dir = df['direction'].values  #6
        Td = df['dewpoint'].values* units.degC    
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
            wind_speed2[j] = float(wind_speed[j])*1.944  # knots
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
           
        # Final catch all
        for l in range(0,len(geopot)-10):
            if ((np.nanmean(geopot[l:l+10]) - geopot[l]) < -100) | \
                ((np.nanmean(geopot[l:l+10]) - geopot[l]) > 250) | \
                ((np.nanmean(p[l:l+10]) -p[l]) < -50) | \
                ((np.nanmean(p[l:l+10]) - p[l]) > 200) | \
                ((p[l]-p[l+1])== 0):
                geopot[l] = np.nan
                u[l] = np.nan
                v[l] = np.nan
                T[l] = np.nan
                Td[l] = np.nan
                p[l] = np.nan
         
        print(len(p),len(Td),len(T))   
            
        # Smooth pressure
        smooth_pres = medfilt(p, 7) * units.hPa
        p = np.array(smooth_pres)
        
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
        
        # Remove nans
        all_data = np.zeros((len(p),7))
        all_data[:,0] = p
        all_data[:,1] = T
        all_data[:,2] = Td
        all_data[:,3] = u
        all_data[:,4] = v
        all_data[:,5] = mpcalc.wind_speed(u,v)
        all_data[:,6] = geopot
        
        all_data = all_data[~np.isnan(all_data).any(axis=1)]
        
        # Plots of u and v with respect to geopotential height
        fig = plt.figure(figsize=(12, 8))
        plt.plot(all_data[:,3],all_data[:,6],c='r',label = 'u')
        plt.plot(all_data[:,4],all_data[:,6],c='k',label = 'v')
        plt.ylim(np.nanmin(all_data[:,6]),np.nanmax(all_data[:,6]))
        plt.xlim(-100,100)
        plt.yticks(fontsize=22)
        plt.xticks(fontsize=22)
        plt.xlabel('u and v components (knots)',fontsize=24,fontweight='bold')
        plt.ylabel('Geopotential Height (m)',fontsize=24,fontweight='bold')
        plt.text(-85,22500,"Date: %s" % date,fontsize=18) 
        plt.legend(loc='upper right', fontsize=22)
        plt.grid()
        plt.title("u and v Components \n Beginning at %s UTC" % time[0],fontsize=25,
                  fontweight='bold')
        launch_time = time[0].replace(':','')
        #plt.savefig('u_v_' + launch_time +'_'+ date + '.png',dpi=300)
        plt.show()
        # Simple 1 minute running mean
        u_1min_rm = np.convolve(all_data[:,3], np.ones(60)/60, mode='valid')
        v_1min_rm = np.convolve(all_data[:,4], np.ones(60)/60, mode='valid')
        geo_1min_rm = np.convolve(all_data[:,6], np.ones(60)/60, mode='valid')
        
        fig = plt.figure(figsize=(12, 8))
        plt.plot(u_1min_rm,geo_1min_rm,zorder=10,c='k', linewidth=2, label='u-comp. ave.')
        plt.scatter(all_data[:,3],all_data[:,6],c='paleturquoise',s=10, label='u-comp')
        plt.xlim(-10,100)
        plt.yticks(fontsize=22)
        plt.xticks(fontsize=22)
        plt.xlabel('u-component (knots)',fontsize=24,fontweight='bold')
        plt.ylabel('Geopotential Height (m)',fontsize=24,fontweight='bold')
        plt.text(70,2500,"Date: %s" % date,fontsize=18) 
        plt.legend(loc='upper right', fontsize=22)
        plt.grid()
        plt.title("u-Component \n Beginning at %s UTC" % time[0],fontsize=25,
                  fontweight='bold')
        plt.show()
        # Difference between 1 min ave and original
        mean_minus_og_u = all_data[0:len(u_1min_rm),3]-u_1min_rm
        mean_minus_og_geo = all_data[0:len(u_1min_rm),6]-geo_1min_rm
        print(mean_minus_og_u) 
        
        # https://realpython.com/linear-regression-in-python/
        # Simple linear regression for u with respect to geopotential height
        model = LinearRegression().fit(all_data[:,3].reshape((-1, 1)), all_data[:,6])
        r_sq = model.score(all_data[:,3].reshape((-1, 1)), all_data[:,6])
        intercept = model.intercept_
        coef = model.coef_
        y_pred = model.intercept_ + \
            np.sum(model.coef_ * all_data[:,3].reshape((-1, 1)),axis=1)
        
        # Simple linear regression for wind speed with respect to geopotential height
        model_ws = LinearRegression().fit(all_data[:,5].reshape((-1, 1)), all_data[:,6])
        r_sq_ws = model_ws.score(all_data[:,5].reshape((-1, 1)), all_data[:,6])
        intercept = model_ws.intercept_
        coef = model_ws.coef_
        y_pred_ws = model_ws.intercept_ + \
            np.sum(model_ws.coef_ * all_data[:,5].reshape((-1, 1)),axis=1)    
    
        # Multiple linear regression
        # (u,v) and height
        u_v = np.zeros((len(all_data),2))
        u_v[:,0] = all_data[:,3]
        u_v[:,1] = all_data[:,4]
        model_m = LinearRegression().fit(u_v, all_data[:,6])
        r_sq_m = model_m.score(u_v, all_data[:,6])
        intercept = model_m.intercept_
        coef_m = model_m.coef_
        y_pred_m = model_m.intercept_ + np.sum(model_m.coef_ * u_v, axis=1)
        
        # Old attempt (works poorly)
        # coef = np.polyfit(uu,geopot2,2)
        # poly1d_fn = np.poly1d(coef) 
        
        # plt.plot(uu, poly1d_fn(uu), '--k')
        # plt.ylabel('Height (m)')
        # plt.title('U-component of Wind and Mean')   
        # plt.savefig('test_u.png',dpi=300)

           
