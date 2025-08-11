'''
The Weather Service Radar 1988 (WSR88D) is a dual doppler radar
with a 10.7 cm (S band) wavelength. The WSR88D scans approximately
14 elevation angles every 5 minutes. The level II data is stored
in a binary format (.ar2v). We begin by importing necessary 
packages, such as pyart, for data analysis
'''

import pyart
import numpy as np
from datetime import datetime
import cartopy.crs as crs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import STORMY

'''
After importing, we must define which NWS radars you would like to use, path to
store the WSR88D files and the time range (start and end) you'd like to grab. We
'''

radar = 'KTYX'
start_time, end_time = datetime(2022, 11, 18, 23,55), datetime(2022, 11, 19, 00, 30)
savepath = "/data2/white/MISC/WSR88D/LVL2/"
WSR88D_path = "/data2/white/MISC/WSR88D/LVL2/"

WSR88D_files = STORMY.download_WSR88D(radar, DateTimeIni=start_time, DateTimeFin=end_time, path_out=WSR88D_path)
print(WSR88D_files)
WSR88D_file = WSR88D_files[0] # Grab the first file for plotting purposes

'''
Now that we have the WSR88D files that fit our time range, we can begin to read the data.
The python package pyart is crucial for reading and plotting radar data, thus is used 
entirely throughout this tutorial. 
'''

radar_obj = pyart.io.read(WSR88D_file) # Create pyart radar object
display_obj = pyart.graph.RadarMapDisplay(radar_obj)  # Create pyart display object

'''
Each level II radar file contains five variables. Here we can define which one we would
like to plot and the plotting settings we should use,
as well as which elevation angle index (e.g. index 0 ~ .5 deg, or lowest angle).
'''
var = "reflectivity" # "reflectivity", "cross_correlation_ratio", "spectrum_width","differential_phase","velocity"
elev_angle = 0

# ZDR colormap
colors1 = plt.cm.binary_r(np.linspace(0.,0.8,33))
colors2= plt.cm.gist_ncar(np.linspace(0.,1,100))
colors = np.vstack((colors1, colors2[10:121]))
ZDR_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

# Dictionary for each variable setting
field_settings = {
    'reflectivity': {
        'vmin': 5,
        'vmax': 70,
        'cmap': 'NWSRef',
        'label': 'Reflectivity',
        'units': 'dBZ',
        },
    'velocity': {
        'vmin': -40,
        'vmax': 40,
        'cmap': 'NWSVel',
        'label': 'Radial Velocity',
        'units': 'm/s',
        },
    'cross_correlation_ratio': {
        'vmin': 0.7,
        'vmax': 1.01,
        'cmap': 'SCook18',
        'label': 'Correlation Coefficient',
        'units': '',
    },
    'differential_reflectivity': {
        'vmin': -2,
        'vmax': 6,
        'cmap': ZDR_cmap,
        'label': 'Differential Reflectivity',
        'units': 'dB',
    },
}

settings = field_settings[var]

'''
Lets use the data we just gathered and variables we defined 
to create a figure of WSR88D data.
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
Now we can plot the data onto the figure as a ppi map. Pyart likes to take full control of the figure, such
as setting a title, grid lines, and colorbar for us. Personally, some of this becomes more of a nuisance 
so I turn many features off and make my own using matplotlib. 
'''

WSR88D_ppi = display_obj.plot_ppi_map(var, mask_outside=True, 
                                                   vmin=settings['vmin'], vmax=settings['vmax'], ax=ax,
                                                   colorbar_flag=True,title_flag=False,add_grid_lines=False,
                                                   cmap=settings['cmap'],colorbar_label=settings['label'])

'''
Nearly complete figure! We now simply add the area we would like to view, colorbar and title. T
he "f" allows use to put variables in strings, such as the time of the radar scan.
'''

extent = [-80, -72, 41, 45] # Min lon, max lon, min lat, max lat
ax.set_extent(extent, crs=crs.PlateCarree())

ax.set_xlabel("Longitude", fontsize=8)
ax.set_ylabel("Lattitude", fontsize=8)

print(WSR88D_file)
time_str, datetime_obj = STORMY.parse_filename_datetime_obs(WSR88D_file)
ax.set_title(radar + " " + settings['label'] + f" at  {datetime_obj}",  fontsize=14, fontweight='bold')

'''
The figure is now complete!! Lets create a suitable filename that we can use to save the figure and
use it in the future. The figure will be saved using savepath, which you defined earlier.
'''

filename = f"WSR88DTUTORIAL_{time_str}.png"
plt.savefig(savepath + filename)
plt.show()

