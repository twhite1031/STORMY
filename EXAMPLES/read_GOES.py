'''
The Geostationary Operational Environmental Satellites (GOES) are a
series of satellites operated by NOAA and NASA. These satellites are equipped with
mulitple tools to analyze the atmosphere. This tutorial will work with netCDF (.nc) files 
folowing the tutorial constructed by @joaohenry23/GOES on Github (All credit goes to him). We begin
by importing necessary packages
'''

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import cartopy.crs as crs
import matplotlib.pyplot as plt
import custom_color_palette as ccp # @jaohenry23 custom package
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import STORMY
import GOES # @jaohenry23 custom package

'''
After importing, we must download the GOES files given a start and end time, as well as
the Satellite and channel(s) we want. Optionally, we can define a path to save the data to.
'''

start_time, end_time = '20221118-135000', '20221118-140100'
savepath = '/data2/white/DATA/MISC/SATELLITE/'
GOES_files = STORMY.download_GOES('goes16', 'ABI-L2-CMIPF',
                      DateTimeIni=start_time, DateTimeFin=end_time,
                      channel=['13'], path_out=savepath)

'''
Now we can read the data and define our viewing extent
'''

extent = [-77, -72, 41, 45] # Min lon, max lon, min lat, max lat
ds = GOES.open_dataset(GOES_files[0]) # Grab the first file to plot

'''
Retrieve the image with coordinates of corners of each
pixel for plotting with pcolormesh
'''

CMI, LonCor, LatCor = ds.image('CMI', lonlat='corner', domain=extent)

'''
Lets retrieve some metadata that we can use for labeling and titles
'''

sat = ds.attribute('platform_ID')
band = ds.variable('band_id').data[0]
wl = ds.variable('band_wavelength').data[0]

'''
Now create a custom colormap with normalization and ticks
based on the typical IR colormaps.
'''

lower_colors = ['maroon','red','darkorange','#ffff00','forestgreen','cyan','royalblue',(148/255,0/255,211/255)]
lower_palette = [lower_colors, ccp.range(180.0,240.0,1.0)]

upper_colors = plt.cm.Greys
upper_palette = [upper_colors, ccp.range(240.0,330.0,1.0), [ccp.range(180.0,330.0,1.0),240.0,330.0]]

cmap, cmticks, norm, bounds = ccp.creates_palette([lower_palette, upper_palette], extend='both')
ticks = ccp.range(180,330,10)

'''
Lets use the data we just gathered to create a figure. First we create a
figure using the PlateCarree Projection focused on our longitude center
based on our extent. fig.add_axes is applied to have more control over
the axes, such as fine tuning labels.
'''

lon_cen = 360.0+(extent[0]+extent[1])/2.0
fig = plt.figure('map', figsize=(4,4), dpi=200)
ax = fig.add_axes([0.1, 0.16, 0.80, 0.75], projection=crs.PlateCarree(lon_cen))

'''
To make a quality figure, we utilize shapefiles that define the borders of lakes, states, countries, 
counties, and more! There are many ways to do this, but I prefer using a helpfer function which
utilizes cartopy.cfeature. Additionally, gridlines are also important to put location and distance 
into perspective, so we will add those as well.
'''

STORMY.add_cartopy_features(ax)

'''
Using the corners of the data we gathered earlier we can now plot the data using pcolormesh.
We also apply our cmap and normalization here.
'''
img = ax.pcolormesh(LonCor.data, LatCor.data, CMI.data, cmap=cmap, norm=norm, transform=crs.PlateCarree())

'''
Similarly to ax creation earlier, we are going to customly place our colorbar. We use our pcolormesh data
to define the cmap.
'''

cb = plt.colorbar(img, ticks=ticks, orientation='horizontal', extend='both',cax=fig.add_axes([0.12, 0.05, 0.76, 0.02]))
cb.ax.tick_params(labelsize=5, labelcolor='black', width=0.5, length=1.5, direction='out', pad=1.0)
cb.set_label(label='{} [{}]'.format(CMI.standard_name, CMI.units), size=5, color='black', weight='normal')
cb.outline.set_linewidth(0.5)

'''
Here we sets 5 evently spaced longitude and latitude tick markers
using the correct format (degree symbol). We also apply labels to each
axis
'''

dx = 5
xticks = np.linspace(extent[0], extent[1], dx)
ax.set_xticks(xticks, crs=crs.PlateCarree())
ax.xaxis.set_major_formatter(LongitudeFormatter(dateline_direction_label=True))
ax.set_xlabel('Longitude', color='black', fontsize=7, labelpad=3.0)

dy = 5
yticks = np.linspace(extent[2], extent[3], dy)
ax.set_yticks(yticks, crs=crs.PlateCarree())
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.set_ylabel('Latitude', color='black', fontsize=7, labelpad=3.0)

'''
Now we can fine tune the areas we would like our ticks and
labels to exist, as well as there sizes and colors. We also will 
define the gridlines and customize its style
'''

ax.tick_params(left=True, right=True, bottom=True, top=True,
               labelleft=True, labelright=False, labelbottom=True, labeltop=False,
               length=0.0, width=0.05, labelsize=5.0, labelcolor='black')

ax.gridlines(xlocs=xticks, ylocs=yticks, alpha=0.6, color='gray',
             draw_labels=False, linewidth=0.25, linestyle='--')

'''
Nearly complete figure! We now simply apply the extent that we set earlier, along 
with a suitable title using the metadata we gathered earlier
'''

ax.set_extent([extent[0]+360.0, extent[1]+360.0, extent[2], extent[3]], crs=crs.PlateCarree())

GOES_time = CMI.time_bounds.data[0].strftime('%Y/%m/%d %H:%M UTC')
ax.set_title('{} - C{:02d} [{:.1f} μm]'.format(sat,band, wl), fontsize=7, loc='left')
ax.set_title(GOES_time, fontsize=7, loc='right')

'''
The figure is now complete!! Lets create a suitable filename that we can use to save the figure and
use it in the future. The figure will be saved using savepath, which you defined earlier.
'''

filename = "GOESTUTORIAL_{GOES_time}.png"
plt.savefig(savepath + filename)

# set the map limits
plt.show()

