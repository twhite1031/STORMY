'''
The Multi Radar Multi Sensor (MRMS) is a system which integrates operational
WSR-88D radars, NWS Soundings, Satellite, and forecast models into algorithms 
that produce many products, such as isothermal reflectivites or quality controled
composite reflectivity in .grib2 format. We being by importing necessary packages.
'''

import cfgrib
import pandas as pd
import numpy as np
from datetime import datetime
import cartopy.crs as crs
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import STORMY

'''
After importing, we must download the MRMS files given a field, start time and endtime, which will search
for MRMS files given the time range. Optionally, we can define a path to save the data to.
'''

field = "Reflectivity_-10C_00.50"
start_time, end_time = datetime(2022, 11, 18, 23,55), datetime(2022, 11, 19, 00, 30)
savepath = '/data2/white/DATA/MISC/MRMS/'

MRMS_files = STORMY.download_MRMS(field= field,start_time=start_time,end_time=end_time,path_out=savepath)
MRMS_file = MRMS_files[0] # Grab the first file for plotting purposes

'''
Now we can read the data and see what time the data is valid for
'''

ds = cfgrib.open_dataset(MRMS_file)
ds = ds.metpy.parse_cf()
 
valid_time = pd.to_datetime(ds['valid_time'].values)
formatted_time = valid_time.strftime("%Y-%m-%d_%H%M")  # Just to the minute
print("Valid datetime:", formatted_time)

'''
Although not necessary for functionality, we choose to rename the 
native data from "unknown" to "reflectivity", something a bit more meaningful
to the user. We also run a dimension check since we can only use 2D data
(unless we slice a specific level) for plan view plotting purposes
'''

# Open and rename for clarity
ds = ds.rename({"unknown": "reflectivity"})
data = ds['reflectivity']  

if data.ndim != 2:
    raise ValueError(f"Incompatible MRMS field: expected 2D, got {data.ndim}D")

'''
The dataset also natively comes with longitude in 0 to 360 format, so we will
account for this by assigning new coords. Additionally, we will subset only a region
of data for plotting, rather than the entire CONUS. This region will also serve
as our viewing extent.
'''

ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))

extent = [-80, -72, 41, 45] # Min lon, max lon, min lat, max lat
lon_min, lon_max = -78.5, -74.5
lat_min, lat_max = 42.5, 44.5

subset = ds.sel(
    longitude=slice(extent[0], extent[1]),
    latitude=slice(extent[3], extent[2])  # Note: latitude is usually decreasing in GRIB
)

'''
Often, we chose to also mask certain values, especially if
they don't mean or indicate much to us. Here we want to only keep
areas of reflectivity > 0
'''

# Mask low reflectivity
reflectivity = subset['reflectivity'].where(subset['reflectivity'] > 0)

'''
Even though we have read the data, we still need to grab the exact values
we will use for plotting. Here we grab the latitude and longitude points, 
as well as the reflectivity values for the data.
'''

# Extract arrays
z = reflectivity.values
MRMS_lats = subset.latitude.values
MRMS_lons = subset.longitude.values

'''
Lets use the data we just gathered to create a figure of MRMS data.
First we create a figure using the PlateCarree Projection
'''

fig = plt.figure(figsize=(10,8))
ax = plt.axes(projection=crs.PlateCarree())

'''
To make a quality figure, we utilize shapefiles that define the borders of lakes, states, countries, 
counties, and more! There are many ways to do this, but I prefer using a helpfer function which
utilizes cartopy.cfeature. Additionally, gridlines are also important to put location and distance 
into perspective, so we will add those as well.
'''

STORMY.add_cartopy_features(ax)
STORMY.format_gridlines(ax, x_inline=False, y_inline=False)

'''
Before we plot the data, we must set levels and create a normalization for the cmap.
The main reason for this is because pcolormesh does not take levels directly.
'''

levels = np.arange(0, 75, 5)  # 0 to 70 dBZ in 5 dBZ bins
cmap = plt.get_cmap("NWSRef")  # or your defined reflectivity colormap
norm = BoundaryNorm(boundaries=levels, ncolors=cmap.N, clip=True)

'''
Now we can plot using pcolormesh, setting our levels in the typical dBZ range for the colormap
'''

mesh = ax.pcolormesh(MRMS_lons, MRMS_lats, z, cmap=cmap,norm=norm,shading="auto", transform=crs.PlateCarree())

'''
Nearly complete figure! We now simply add a colorbar and title. The .format allows use to put variables as string
inside {}, such as the units for the MRMS data. The "f" allows use to put variables in strings, such
as the time we are using for the MRMS data.
'''

cb = fig.colorbar(mesh, ticks=levels, orientation='horizontal', shrink=.8,ax=ax)
cb.set_label(label='{}'.format("dBZ"), color='black')
cb.outline.set_linewidth(0.5)

ax.set_title(f"MRMS -10C Reflectivity at {valid_time}", fontsize=14)

'''
The figure is now complete!! Lets create a suitable filename that we can use to save the figure and
use it in the future. The figure will be saved using savepath, which you defined earlier.
'''

filename = "MRMSTUTORIAL_{formatted_time}.png"
plt.savefig(savepath + filename)
plt.show()
