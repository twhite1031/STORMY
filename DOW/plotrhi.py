import os
import warnings
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pyart
from pyart.testing import get_test_data
from metpy.plots import USCOUNTIES
from metpy.calc import dewpoint_from_relative_humidity
from metpy.units import units
import xarray
import cartopy.crs as ccrs
import cartopy.feature as cfeature
warnings.filterwarnings('ignore')
#---------------------------------------------

# Adjust path to file and filename here
path = "/data1/barber/SOURCE/20240402_IOP01/DOW6/radar/cfradial_swps/"
#cfrad.20240402_193858.154_DOW6_v186_s00_az359.00_RHI
file = "cfrad.20240402_210520.827_DOW6_v234_s00_az147.00_RHI.nc"

# Read in the file
radar = pyart.io.read(path + file)
radar

#Create a figure
fig = plt.figure(figsize=(30,15))


# Create the Radar Display Object
display = pyart.graph.RadarMapDisplay(radar)

# Plot the RHI, call the variable
display.plot_rhi('WIDTH', norm=None,cmap="Spectral_r")
plt.ylim(0,10)
plt.show()

