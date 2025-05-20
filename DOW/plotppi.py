import os
import warnings
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pyart
from pyart.testing import get_test_data
from metpy.plots import USCOUNTIES
from metpy.calc import dewpoint_from_relative_humidity
from metpy.units import units
import cartopy.crs as ccrs
import cartopy.feature as cfeature
warnings.filterwarnings('ignore')
#---------------------------------------------

path = "/data1/barber/SOURCE/20240402_IOP01/DOW6/radar/cfradial_swps/"
file = "cfrad.20240402_213143.555_DOW6_v259_s26_el8.02_SUR.nc"
radar = pyart.io.read(path + file)
radar

bounds = [42.0, 44.5, -75.5, -78]

fig = plt.figure(figsize=[12, 12])
ax1 = fig.add_subplot(111, projection=ccrs.PlateCarree())
#ax1 = fig.add_subplot(111)

ax1.add_feature(cfeature.STATES, linewidth=3)
ax1.add_feature(USCOUNTIES, alpha=0.4)

# Create the Radar Display Object
display = pyart.graph.RadarMapDisplay(radar)

# Plot the reflectivty
#vmin = -20, vmax = 40 for dbz
display.plot_ppi_map('ZDRM',ax=ax1,min_lat=bounds[0],max_lat=bounds[1],min_lon=bounds[2],max_lon=bounds[3],
                     embellish=False, norm=None,cmap="Spectral_r")


plt.ylim(0, 20) 
plt.show()
