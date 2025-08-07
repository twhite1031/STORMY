'''
The Lightning Mapping Array (LMA) is a ground-based observational 
system that detects and locates very high frequency (VHF) radio emissions produced
lightning discharges. National Severe Storms Labratory (NSSL) operates and stores LMA
data in .h5 and .gz file formats. This tutorial will show how to read .h5 LMA files.
We being by importing necessary packages.
'''

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import cartopy.crs as crs
import matplotlib.pyplot as plt
import STORMY

'''
After importing, we must download the LMA files given a time and a buffer, which will search
for LMA files time +- buffer. Optionally, we can define a path to save the data to.
'''

time = datetime(2022, 11, 18, 23,55)
tbuffer = 1800 # seconds
savepath = "/data2/white/DATA/MISC/LMA"

LMA_files = STORMY.download_LMA(time,tbuffer=tbuffer, path_out=savepath)

'''
Before reading the data, we must first define the criteria of the LMA data. That is,
what VHF points meet a certain number of station, or points per flash (etc.)
'''

min_events_per_flash = 10
min_stations = 6
max_chi = 1 # Lower = more confident it's a good solution

'''
Within each LMA_file we must parse the correct timestamp to construct the correct HDF5 
group name (flash/flash events) which is inside the file. We will read each file and store flashes 
and flash events (VHF Sources). We also must apply an offset to prevent files from overwriting eachother
and create an absolute time column instead of seconds from midnight
'''

flashes = pd.DataFrame() # Initialize DataFrame to store the data
flash_events = pd.DataFrame() # Initialize DataFrame to store the data

for filename in LMA_files:
    timeobj = datetime.strptime(
        filename.split('/')[-1],
        "LYLOUT_%y%m%d_%H%M%S_0600.dat.flash.h5"
    )
    midnight = datetime(*timeobj.timetuple()[:3])  # Midnight of that file's day
    time_str = timeobj.strftime('%y%m%d_%H%M') # Format for subdirectories
    
    f2 = pd.read_hdf(filename, f'flashes/LMA_{time_str}00_600')
    e2 = pd.read_hdf(filename, f'events/LMA_{time_str}00_600')

    # Add absolute time column to the events (Instead of seconds from midnight)
    f2["datetime"] = [midnight + timedelta(seconds=s) for s in f2.start]
    e2["datetime"] = [midnight + timedelta(seconds=s) for s in e2.time]

    # Construct a list that does not overwrite eachother
    if not flashes.empty:
        offset = flashes.flash_id.max() + 1
        f2.flash_id += offset
        e2.flash_id += offset
    
    # Add flash and flash events to original dateframe
    flashes = pd.concat([flashes, f2], ignore_index=True)
    flash_events = pd.concat([flash_events, e2], ignore_index=True)

'''
With our completed DataFrames we can now start to create our filter. We are mainly
interested in filtering the flash events given our time, time buffer, max chi^2 and
minimum stations
'''

selection = (
    (flash_events["datetime"] >= time) &
    (flash_events["datetime"] < time + timedelta(seconds=tbuffer)) &
    (flash_events.chi2 <= max_chi) &
    (flash_events.stations >= min_stations)
)

'''
Now we apply our filter to any group within flash_events. Here we will grab
the lattitude and longitude since that is what we desire for plotting purposes
'''

lma_lon = flash_events.lon[selection].values
lma_lat = flash_events.lat[selection].values

'''
Lets use the data we just gathered to create a figure of filtered LMA points.
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
Now we can plot using ax.scatter, adding a legend as well
'''

ax.scatter(lma_lon, lma_lat,color='fuchsia',s=15,marker='^',label='LMA Events',transform=crs.PlateCarree())
ax.legend(loc='upper right')

'''
Often we are going to want to give a specific viewing area to get the best plot.
Here we define a bounding box and apply it to the figure as well as setting a title and labels
'''

extent = [-76.5, -75.8, 43.9, 44.2] # Min lon, max lon, min lat, max lat
ax.set_extent(extent, crs=crs.PlateCarree())
ax.set_title("Filtered LMA Lightning Events")

'''
The figure is now complete!! Lets create a suitable filename that we can use to save the figure and
use it in the future. The figure will be saved using savepath, which you defined earlier.
'''

time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
filename = "LMATUTORIAL_{time_str}.png"
plt.savefig(savepath + filename)
plt.show()

