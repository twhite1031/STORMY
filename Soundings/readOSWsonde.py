import pandas as pd
from metpy.units import units
import metpy.calc as mpcalc
from metpy.plots import SkewT
import matplotlib.pyplot as plt
import wrf
from netCDF4 import Dataset
import numpy as np
from datetime import datetime, timedelta
import os 

# User input for file
year = 2022
month = 11
day = 19
hour = "00"
minute = "00"
domain = 2
timeidx = 0
IOP = 2
ATTEMPT ="1"
#Oswego is 99 and 211
gridboxlat=99
gridboxlon=211
savefig = True

# Define the column names based on the data structure
column_names = ['Time', 'Height', 'Pressure', 'Temperature', 'Relative_Humidity', 'Wind_Direction', 'Wind_Speed', 'Range', 'Latitude', 'Longitude']

# Read the data into a DataFrame
file_path = "/data2/white/DATA/PROJ_LEE/IOP_2/SOUNDINGS/edt_20221118_2357.txt"
df = pd.read_csv(file_path, delim_whitespace=True, header=None, names=column_names)

# Convert the necessary columns to appropriate units
p  = df['Pressure'].values * units.hPa
temp  = df['Temperature'].values * units.degC
wspd = df['Wind_Speed'].values * units.meter / units.second
wdir = df['Wind_Direction'].values * units.deg
rh = df['Relative_Humidity'].values * units.percent
dewtemp = mpcalc.dewpoint_from_relative_humidity(temp, rh)

u_v = mpcalc.wind_components(wspd, wdir)
u = u_v[0]
v = u_v[1]

# Create a figure with two subplots
fig = plt.figure(figsize=(12, 6))
# Example of defining your own vertical barb spacing
skew = SkewT(fig=fig, subplot=(1,2,1))

# Plot the data using normal plotting functions, in this case using
# log scaling in Y, as dictated by the typical meteorological plot
skew.plot(p, temp, 'r')
skew.plot(p, dewtemp, 'g')

# Plot only values nearest to defined interval values
skew.plot_barbs(p[::50], u[::50], v[::50])

# Add the relevant special lines
skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines()
skew.ax.set_ylim(1000, 700)
skew.ax.set_xlim(-40, 20)
skew.ax.set_xlabel('Temperature ($^\circ$C)')
skew.ax.set_ylabel('Pressure (hPa)')

#plt.title(f"SkewT at 20221118_2357 (Shineman Roof) ")
#if savefig == True:
#        plt.savefig(path+f"skewT{year}{month}{day}{hour}{minute}A{ATTEMPT}.png")

# Open the NetCDF file
path = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_{IOP}/ATTEMPT_{ATTEMPT}/"
pattern = f"wrfout_d0{domain}_{year}-{month}-{day}_{hour}:{minute}:00"
wrfin = Dataset(path+pattern)

# Convert desired coorindates to WRF gridbox coordinates
#x_y = wrf.ll_to_xy(wrfin, lat_lon[0], lat_lon[1])

# Read skewT variables in
p1 = wrf.getvar(wrfin,"pressure",timeidx=timeidx)
T1 = wrf.getvar(wrfin,"tc",timeidx=timeidx)
Td1 = wrf.getvar(wrfin,"td",timeidx=timeidx)
u1 = wrf.getvar(wrfin,"ua",timeidx=timeidx)
v1 = wrf.getvar(wrfin,"va",timeidx=timeidx)

# Get variables for desired coordinates
p = p1[:,gridboxlat,gridboxlon] * units.hPa
T = T1[:,gridboxlat,gridboxlon] * units.degC
Td = Td1[:,gridboxlat,gridboxlon] * units.degC
u = u1[:,gridboxlat,gridboxlon] * units('kt')
v = v1[:,gridboxlat,gridboxlon] * units('kt')

#Test if the coordinates are correct
lat1 = wrf.getvar(wrfin,"lat",timeidx=timeidx)
lon1 = wrf.getvar(wrfin,"lon",timeidx=timeidx)
lat = lat1[gridboxlat,gridboxlon] * units.degree_north
lon = lon1[gridboxlat,gridboxlon] * units.degree_east
print(lat)
print(lon) 

# Example of defining your own vertical barb spacing
skew1 = SkewT(fig=fig,subplot=(1,2,2))

# Plot the data using normal plotting functions, in this case using
# log scaling in Y, as dictated by the typical meteorological plot
skew1.plot(p, T, 'r')
skew1.plot(p, Td, 'g')

# Set spacing interval--Every 50 mb from 1000 to 100 mb
my_interval = np.arange(400, 1000, 50) * units('mbar')

# Get indexes of values closest to defined interval
ix = mpcalc.resample_nn_1d(p, my_interval)

# Plot only values nearest to defined interval values
skew1.plot_barbs(p[ix], u[ix], v[ix])

# Add the relevant special lines
skew1.plot_dry_adiabats()
skew1.plot_moist_adiabats()
skew1.plot_mixing_lines()
skew1.ax.set_ylim(1000, 700)
skew1.ax.set_xlim(-40, 20)
skew1.ax.set_xlabel('Temperature ($^\circ$C)')
skew1.ax.set_ylabel('Pressure (hPa)')

# Define the format of the datetime string in your filename
datetime_format = "wrfout_d02_%Y-%m-%d_%H:%M:%S"

# Parse the datetime string into a datetime object
time_object = datetime.strptime(os.path.basename(path+pattern), datetime_format)

# Add timeidx value
add_time = 5 * timeidx
time_object_adjusted = time_object + timedelta(minutes=add_time)

skew.ax.set_title(f"Observed skewT at 2022-11-18 23:57:00")
skew1.ax.set_title(f"WRF Simulated skewT at " + str(time_object_adjusted))


if savefig == True:
        plt.savefig(path+f"OSWskewT{year}{month}{day}{hour}{minute}A{ATTEMPT}.png")
plt.show()

                       
