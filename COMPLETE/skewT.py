import wrf
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import metpy.calc as mpcalc
from metpy.plots import SkewT
from metpy.units import units
import wrffuncs
from datetime import datetime, timedelta
import pandas as pd
"""
A standard script to make a skewT given a lat/lon point which finds 
the nearest gridbox to that point
"""

# --- USER INPUT ---

wrf_date_time = datetime(1997,1,12,1,52,00)
domain = 2
lat_lon = [43.598, -75.918]

SIMULATION = "NORMAL" # If comparing runs
path = f"/data2/white/WRF_OUTPUTS/SEMINAR/{SIMULATION}_ATTEMPT/"

# --- END USER INPUT ---
time_df = wrffuncs.build_time_df(path, domain)
obs_time = pd.to_datetime(wrf_date_time)

# Compute absolute time difference
closest_idx = (time_df["time"] - obs_time).abs().argmin()

# Extract the matched row
match = time_df.iloc[closest_idx]

# Unpack matched file info
matched_file = match["filename"]
matched_timeidx = match["timeidx"]
matched_time = match["time"]

print(f"Closest match: {matched_time} in file {matched_file} at time index {matched_timeidx}")

with Dataset(matched_file) as ds:
    # Convert desired coorindates to WRF gridbox coordinates
    x_y = wrf.ll_to_xy(ds, lat_lon[0], lat_lon[1])

    # Read skewT variables in
    p1 = wrf.getvar(ds,"pressure",timeidx=matched_timeidx)
    T1 = wrf.getvar(ds,"tc",timeidx=matched_timeidx)
    Td1 = wrf.getvar(ds,"td",timeidx=matched_timeidx)
    u1 = wrf.getvar(ds,"ua",timeidx=matched_timeidx)
    v1 = wrf.getvar(ds,"va",timeidx=matched_timeidx)

# Get variables for desired coordinates
p = p1[:,x_y[1],x_y[0]] * units.hPa
T = T1[:,x_y[1],x_y[0]] * units.degC
Td = Td1[:,x_y[1],x_y[0]] * units.degC
u = u1[:,x_y[1],x_y[0]] * units('kt')
v = v1[:,x_y[1],x_y[0]] * units('kt')

#Test if the coordinates are correct
lat1 = wrf.getvar(Dataset(matched_file),"lat",timeidx=matched_timeidx)
lon1 = wrf.getvar(Dataset(matched_file),"lon",timeidx=matched_timeidx)
lat = lat1[x_y[1],x_y[0]] * units.degree_north
lon = lon1[x_y[1],x_y[0]] * units.degree_east
print(lat)
print(lon) 

# Example of defining your own vertical barb spacing
skew = SkewT()

# Plot the data using normal plotting functions, in this case using
# log scaling in Y, as dictated by the typical meteorological plot
skew.plot(p, T, 'r')
skew.plot(p, Td, 'g')

# Set spacing interval--Every 50 mb from 1000 to 100 mb
my_interval = np.arange(100, 1000, 50) * units('mbar')

# Get indexes of values closest to defined interval
ix = mpcalc.resample_nn_1d(p, my_interval)

# Plot only values nearest to defined interval values
skew.plot_barbs(p[ix], u[ix], v[ix])

# Add the relevant special lines
skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines()
skew.ax.set_ylim(1000, 100)
skew.ax.set_xlim(-60, 40)
skew.ax.set_xlabel('Temperature ($^\circ$C)')
skew.ax.set_ylabel('Pressure (hPa)')

date_format = wrf_date_time.strftime("%Y-%m-%d %H:%M:%S")
plt.title(f"SkewT at {date_format}")
plt.show()

