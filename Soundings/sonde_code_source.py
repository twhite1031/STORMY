import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from metpy.calc import parcel_profile, most_unstable_parcel, dewpoint, most_unstable_cape_cin, wind_components
from metpy.plots import SkewT
from metpy.units import units
import os, glob, sys

indir = "/data1/white/Downloads/SOURCE/sondedata/"

filelist = sorted(glob.glob(f'{indir}/SOURCE06*.txt'))
nfiles = len(filelist)
col_names = ['time', 'pressure', 'temperature', 'relative_humidity',
             'speed', 'direction', 'geopot', 'dewpoint']

for i in range(0,nfiles):
    df = pd.read_csv(filelist[i],
                 encoding='latin', header=18,
                 delim_whitespace=True,
                 usecols=[1, 2, 3, 4, 5, 6, 10, 12],
                 names=col_names)

# Replace '-' with np.NaN
df = df.replace('-', np.NaN)
date = filelist[i][75:81]

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
u, v = wind_components(wind_speed22, wind_dir22)

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

# Calculate the most unstable parcel
#where_p_200 = np.where(p < 200)
#mup = most_unstable_parcel(p[0:where_p_200[0][0]]*units.hPa, 
#                                T[0:where_p_200[0][0]]*units.degC, 
#                                Td[0:where_p_200[0][0]]*units.degC)

#prof = parcel_profile(p[0:where_p_200[0][0]]*units.hPa, T[0]*units.degC, 
#                      Td[0]*units.degC).to('degC')

#mu_cape, mu_cin = most_unstable_cape_cin(p*units.hPa,T*units.degC,Td*units.degC)

# Create a new figure
fig = plt.figure(figsize=(9, 9))

# Create a skew-T axis
skew = SkewT(fig)

# Plot the data
skew.plot(p, T, 'r', linewidth=2, label='Temperature')
skew.plot(p, Td, 'g', linewidth=2, label='Dew Point')
#skew.plot(np.array(mup[0]),np.array(mup[1]),'ko', markerfacecolor='black')
#skew.plot(p[0:where_p_200[0][0]],prof,'k')
skew.plot_barbs(p[0::120], u[0::120], v[0::120])
skew.ax.set_ylim(1000, 25)
skew.ax.set_xlim(-40, 60)

skew.ax.set_yticks([1000, 900, 800, 700, 600, 500,
                    400, 300, 200, 100, 50, 25])
skew.ax.set_xticks([-60, -50,-40,-30,-20,-10,0,10,20,30,40])

skew.ax.xaxis.set_tick_params(which='major', labelsize=16, direction='out')
skew.ax.yaxis.set_tick_params(which='major', labelsize=16, direction='out')
skew.ax.set_ylabel('Pressure (hPa)', fontsize=24,fontweight='bold')
skew.ax.set_xlabel('Temperature ($^\circ$C)', fontsize=24,fontweight='bold')
skew.ax.set_title("Balloon Launch at %s UTC" %
                  time[0], fontsize=26, fontweight='bold')

props = dict(boxstyle='round', facecolor='white', alpha=1)
skew.ax.text(0.05, 0.95, 'Date: %s' % date, transform=skew.ax.transAxes,
              fontsize=18, verticalalignment='top',bbox=props)

# Add CAPE and CIN to the plot
skew.plot_mixing_lines()
#skew.shade_cape(p[0:where_p_200[0][0]], T[0:where_p_200[0][0]], np.array(prof))
skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines()

launch_time = time[0].replace(':','')
#plt.savefig('sonde_' + launch_time +'_'+ date + '.png',dpi=300)
plt.show()    

