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
day = 18
hour = "23"
minute = "00"
domain = 2
timeidx = 0
IOP = 2
ATTEMPT ="1"

# Define the column names based on the data structure
column_names = ['Pressure', 'Height', 'Temperature', 'Dewpoint_Temperature', 'Relative_Humidity','Mixing_Ratio', 'Wind_Direction', 'Wind_Speed', 'Theta_A', 'Theta_E', 'Theta_V']

# Read the data into a DataFrame
file_path = "/data2/white/DATA/PROJ_LEE/IOP_2/SOUNDINGS/MW41_output_textFormat_20221118_225843.txt"
df = pd.read_csv(file_path, delim_whitespace=True, header=6, names=column_names)

# Convert the necessary columns to appropriate units
p  = df['Pressure'].values * units.hPa
for idx,item in enumerate(p):
    if(item.magnitude < float(p[idx+1].magnitude)):
        index_of_pop = idx+1
        print(p[idx+1])
        break
    

print("Balloon popped at: ", index_of_pop)
p = p[0:index_of_pop]
temp  = df['Temperature'].values[0:index_of_pop] * units.degC
wspd = df['Wind_Speed'].values[0:index_of_pop] * units.meter / units.second
wdir = df['Wind_Direction'].values[0:index_of_pop] * units.deg
rh = df['Relative_Humidity'].values[0:index_of_pop] * units.percent
dewtemp = mpcalc.dewpoint_from_relative_humidity(temp, rh)
    

print("Balloon popped at: ", index_of_pop)
u_v = mpcalc.wind_components(wspd, wdir)
u = u_v[0]
v = u_v[1]

fig = plt.figure(figsize=(12,6))
# Example of defining your own vertical barb spacing
skew = SkewT(fig=fig, rotation=45)

# Plot the data using normal plotting functions, in this case using
# log scaling in Y, as dictated by the typical meteorological plot
skew.plot(p, temp, 'r', label="Observed Temperature")
skew.plot(p, dewtemp, 'g',label="Observed Dewpoint")

# Plot only values nearest to defined interval values
skew.plot_barbs(p[::75], u[::75], v[::75])

# Add the relevant special lines
skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines()
skew.ax.set_ylim(1000, 400)
skew.ax.set_xlim(-40, 20)
skew.ax.set_xlabel('Temperature ($^\circ$C)')
skew.ax.set_ylabel('Pressure (hPa)')

plt.title(f"SkewT at 20221118_225843 (Henderson Harbor) ")
#plt.savefig()

# Open the NetCDF file
path = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_{IOP}/ATTEMPT_{ATTEMPT}/"
pattern = f"wrfout_d0{domain}_{year}-{month}-{day}_{hour}:{minute}:00"
wrfin = Dataset(path+pattern)

# Convert desired coorindates to WRF gridbox coordinates
x_y = wrf.ll_to_xy(wrfin, 43.8883, -76.1567)

# Read skewT variables in
p1 = wrf.getvar(wrfin,"pressure",timeidx=timeidx)
T1 = wrf.getvar(wrfin,"tc",timeidx=timeidx)
Td1 = wrf.getvar(wrfin,"td",timeidx=timeidx)
u1 = wrf.getvar(wrfin,"ua",timeidx=timeidx,units="kt")
v1 = wrf.getvar(wrfin,"va",timeidx=timeidx,units="kt")

# Get variables for desired coordinates
p = p1[:,x_y[1],x_y[0]] * units.hPa
T = T1[:,x_y[1],x_y[0]] * units.degC
Td = Td1[:,x_y[1],x_y[0]] * units.degC
u = u1[:,x_y[1],x_y[0]] * units('kt')
v = v1[:,x_y[1],x_y[0]] * units('kt')

#Test if the coordinates are correct
lat1 = wrf.getvar(wrfin,"lat",timeidx=timeidx)
lon1 = wrf.getvar(wrfin,"lon",timeidx=timeidx)
lat = lat1[x_y[1],x_y[0]] * units.degree_north
lon = lon1[x_y[1],x_y[0]] * units.degree_east
print(lat)
print(lon) 

# Plot the data using normal plotting functions, in this case using
# log scaling in Y, as dictated by the typical meteorological plot

skew.plot(p, T, '#FF4500', label="WRF Temperature")
skew.plot(p, Td, '#66efc4',label="WRF Dewpoint")

# Set spacing interval--Every 50 mb from 1000 to 100 mb
my_interval = np.arange(400, 1000, 50) * units('mbar')

# Get indexes of values closest to defined interval
ix = mpcalc.resample_nn_1d(p, my_interval)
offset=.05
# Plot only values nearest to defined interval values
skew.plot_barbs(wrf.to_np(p[ix]), u[ix], v[ix],xloc=1.0+offset)

# Add the relevant special lines
skew.ax.legend()
# Define the format of the datetime string in your filename
datetime_format = "wrfout_d02_%Y-%m-%d_%H:%M:%S"

# Parse the datetime string into a datetime object
time_object = datetime.strptime(os.path.basename(path+pattern), datetime_format)

# Add timeidx value
add_time = 5 * timeidx
time_object_adjusted = time_object + timedelta(minutes=add_time)


#if savefig == True:
      # plt.savefig(path+f"OSWskewT{year}{month}{day}{hour}{minute}A{ATTEMPT}.png")
plt.show()
                       
