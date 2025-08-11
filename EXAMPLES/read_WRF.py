'''
The Weather Research and Forecasting (WRF) model is a non-hydrostatic
using terrain-following hydrostatic-pressure veritcal coordinates.
Model output is in netCDF (.nc) format. We begin by importing necessary 
packages, such as wrf-python, for data analysis
'''

from wrf import (getvar, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords, to_np) 
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from datetime import datetime
import cartopy.crs as crs
import matplotlib.pyplot as plt
import STORMY

'''
After importing, we must define which domain you would like to work with and the time you'd like to grab. 
This is completely dependent on your model files; however, the time will adjust to the available
times based on your model run. You should also define your path to ***ALL***  WRF files and where
you'd like to save figures.
'''

wrf_date_time = datetime(2022, 11, 18, 13, 50)
domain = 2

WRF_path = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_1/"
savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/ATTEMPT_1/"

'''
Now that we have the necessary paths, model time, and model domain we can construct a
time dataframe which we will use to find the closest model flie to your set time.
'''

time_df = STORMY.build_time_df(WRF_path, domain)
obs_time = pd.to_datetime(wrf_date_time) # Convert datetime to pandas datetime
closest_idx = (time_df["time"] - obs_time).abs().argmin() # Closest index that matches your time
match = time_df.iloc[closest_idx] # Use the index to find the time

# Unpack matched file info, which includes the filepath (str), timeidx (int), and time (datetime)
matched_file = match["filename"]
matched_timeidx = match["timeidx"]
matched_time = match["time"]
print(f"Closest match: {matched_time} in file {matched_file} at time index {matched_timeidx}")

'''
Given the file and timeidx we can now read in a WRF variable using netCDF to an xarray.DataArray. These variables can either be 2D or 3D, 
depending on if the variable is distributed throughout each height level. We will read in T2, which is
the modeled two meter temperature and can be plotted directly on a plan view map
'''

with Dataset(matched_file) as ds:
    T2 = getvar(ds, "T2", timeidx = matched_timeidx)

# Lets read some metadata about the variable
print("Type: ", type(T2))
print("Shape:", T2.shape)
print("Dimensions:", T2.dims)
print("Coordinates:", T2.coords)
print("Attributes:", T2.attrs)
print("Units:", T2.attrs.get('units', 'No units found'))

'''
Now that we have the data in a useable format, we can prepare to create a visually
appealing figure. We will need to grab the projection information, lat/lon coordinates
which matches the data and the x/y limits for the plot
'''

lats, lons = latlon_coords(T2)
cart_proj = get_cartopy(T2)
WRF_ylim = cartopy_ylim(T2)
WRF_xlim = cartopy_xlim(T2)

'''
Lets create our figure now and apply our projection 
'''

fig = plt.figure(figsize=(16,12))
ax = plt.axes(projection=cart_proj)

'''
To make a quality figure, we utilize shapefiles that define the borders of lakes, states, countries, 
counties, and more! There are many ways to do this, but I prefer using a helpfer function which
utilizes cartopy.cfeature. Additionally, gridlines are also important to put location and distance 
into perspective, so we will add those as well.
'''

STORMY.add_cartopy_features(ax)
STORMY.format_gridlines(ax, x_inline=False, y_inline=False, xpadding=20, ypadding=20)

'''
It's now time to define our contour map levels for our data and plot filled contours
onto the figure. We will use a STORMY helper function and convert our lat/lon to numpy arrays (to_np).
'''

T2_levels = STORMY.make_contour_levels(T2, interval=1)
T2_contours = plt.contourf(to_np(lons), to_np(lats), T2, levels=T2_levels, cmap ="hot_r", transform = crs.PlateCarree())

'''
Nearly complete figure! We now simply add a colorbar and title. The "f" allows use to put variables in strings, such
as the time we are using for the WRF run.
'''

cbar = plt.colorbar()
cbar.set_label(T2.attrs.get("units")) # Lets label the colorbar with the units we are using
plt.title(f"T2 at {matched_time}", fontsize=14)

'''
The figure is now complete!! Lets create a suitable filename that we can use to save the figure and
use it in the future. The figure will be saved using savepath, which you defined earlier.
'''

time_str = matched_time.strftime("%Y-%m-%d_%H-%M-%S")
filename = f"WRFTUTORIAL_{time_str}.png"
plt.savefig(savepath + filename)
plt.show()




