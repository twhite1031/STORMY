import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature
from netCDF4 import Dataset
from metpy.plots import ctables
from wrf import (getvar, to_np, get_cartopy, latlon_coords, cartopy_xlim, cartopy_ylim)
import pyart
from matplotlib.colors import from_levels_and_colors
import cartopy.io.shapereader as shpreader
from pyxlma.lmalib.io import read as lma_read


# User input for file
year = 2022
month = 11
day = 19
hour = "00"
minute = "00"
domain = 2
timeidx = 0
IOP = 2
ATTEMPT = "1B"
savefig = True

height = 850

# Open the NetCDF file
path = f"/data1/white/WRF_OUTPUTS/PROJ_LEE/IOP_2/ATTEMPT_1B/"
pattern = f"wrfout_d0{domain}_{year}-{month}-{day}_{hour}:{minute}:00"
ncfile = Dataset(path+pattern)

# Open the LMA NetCDF file
lmapath = "/data1/white/Downloads/PROJ_LEE/LMADATA/" 
lmafilename = "LYLOUT_221119_001000_0600.dat.gz"
ds, starttime  = lma_read.dataset(lmapath + lmafilename)


# Get the WRF variables
mdbz = getvar(ncfile, "mdbz")
lpi = getvar(ncfile, "LPI")

# Get the observed variables
radar_data = "/data1/white/PYTHON_SCRIPTS/PROJ_LEE/KTYX20221119_000109_V06.ar2v"
obs_dbz = pyart.io.read_nexrad_archive(radar_data)
display = pyart.graph.RadarMapDisplay(obs_dbz)

# Use these lines to see the fields
#radar_fields = obs_dbz.fields
#print("Observed radar fields: ", radar_fields)

# Get the lat/lon points
lats, lons = latlon_coords(mdbz)

# Get the cartopy projection object
cart_proj = get_cartopy(mdbz)

# Create a figure that will have 3 subplots
fig = plt.figure(figsize=(30,15))
ax_wspd = fig.add_subplot(1,2,1, projection=cart_proj)
ax_dbz = fig.add_subplot(1,2,2, projection=cart_proj)

# Set the margins to 0
ax_wspd.margins(x=0,y=0,tight=True)
ax_dbz.margins(x=0,y=0,tight=True)

# Special stuff for counties
reader = shpreader.Reader('countyl010g.shp')
counties = list(reader.geometries())
COUNTIES = cfeature.ShapelyFeature(counties, crs.PlateCarree(),zorder=5)


# Draw the county lines
ax_wspd.add_feature(COUNTIES, facecolor='none', edgecolor='black',linewidth=1)
ax_dbz.add_feature(COUNTIES, facecolor='none', edgecolor='black',linewidth=1)

ax_dbz.set_xlim(cartopy_xlim(mdbz))
ax_dbz.set_ylim(cartopy_ylim(mdbz))

gl = ax_wspd.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=False)
gl.xlabel_style = {'rotation': 'horizontal','size': 14,'ha':'center'} # Change 14 to your desired font size
gl.ylabel_style = {'size': 14}  # Change 14 to your desired font size
gl.xlines = True
gl.ylines = True
gl.top_labels = False  # Disable top labels
gl.right_labels = False  # Disable right labels
gl.xpadding = 20

gl2 = ax_dbz.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=False)
gl2.xlabel_style = {'rotation': 'horizontal','size': 14,'ha':'center'} # Change 14 to your desired font size
gl2.ylabel_style = {'size': 14}  # Change 14 to your desired font size
gl2.xlines = True
gl2.ylines = True
gl2.top_labels = False  # Disable top labels
gl2.right_labels = False  # Disable right labels
gl2.xpadding = 20

levels = [10, 15, 20, 25, 30, 35, 40, 45,50,55,60]

nwscmap = ctables.registry.get_colortable('NWSReflectivity')
mdbz = np.ma.masked_outside(to_np(mdbz),10,65)
mdbz_contourline = ax_wspd.contour(to_np(lons), to_np(lats), mdbz,levels=levels,vmin=10,vmax=60,cmap=nwscmap, transform=crs.PlateCarree(),zorder=2)

# This line will plot inline labels from contour lines
#ax_dbz.clabel(mdbz_contourline, inline=True, fontsize=4,fmt='%.2f')

# This is used to get the correct colorbar
mdbz_contour = ax_wspd.contourf(to_np(lons), to_np(lats), mdbz,levels=levels,vmin=10,vmax=60,cmap=nwscmap, transform=crs.PlateCarree(),zorder=0)

# CMap for LPI
dbz_levels = np.arange(5., 75., 5.)
dbz_rgb = np.array([[255,255,255],[255,255,255],
                    [3,44,244], [3,0,210],
                    [2,253,2], [1,197,1],
                    [0,142,0], [253,248,2],
                    [229,188,0], [253,149,0],
                    [253,0,0], [212,0,0],
                    [188,0,0],[248,0,253]], np.float32) / 255.
dbz_map, dbz_norm = from_levels_and_colors(dbz_levels, dbz_rgb,extend="max")

# Add LPI to the plot with simulated reflectivity
# Make the filled countours with specified levels and range
lpi_levels = np.arange(0,6,1)
lpi_contour = ax_wspd.contourf(to_np(lons), to_np(lats),to_np(lpi),levels=lpi_levels, transform=crs.PlateCarree(),cmap=dbz_map)


comp_ref = pyart.retrieve.composite_reflectivity(obs_dbz, field="reflectivity")
display = pyart.graph.RadarMapDisplay(comp_ref)

obs_contour = display.plot_ppi_map("composite_reflectivity",vmin=10,vmax=60,mask_outside=True,ax=ax_dbz, colorbar_flag=False, title_flag=False, add_grid_lines=False, cmap=nwscmap,zorder=2)

# Plot LMA event locations
#count_subset = {'number_of_events':slice(0,10,1)}

station_filter = (ds.event_stations >= 7)
chi_filter = (ds.event_chi2 <= 1.0)

filter_subset = {'number_of_events':(chi_filter & station_filter)}
# Use this line to filter number of events by slices
#filter1 = ds[filter_subset].isel({'number_of_events':slice(0,10)})

# note that we first filter on all the data select 10000 points, and then on that dataset we further filter 
art = ds[filter_subset].plot.scatter(x='event_longitude', y='event_latitude',c='black', ax=ax_wspd, 
                    s=10, vmin=0.0,edgecolor='black',vmax=5000,transform=crs.PlateCarree())

# KART flash observation
ax_wspd.scatter([-76.023498],[43.990428],c='red',s=75,marker="X",transform=crs.PlateCarree())

## Create a color bar for both plots
fig.subplots_adjust(right=0.8)

# Add the colorbar for the first plot
cbar_ax1 = fig.add_axes([ax_dbz.get_position().x1 + 0.07,
                         ax_dbz.get_position().y0,
                         0.02,
                         ax_dbz.get_position().height])
cbar1 = fig.colorbar(mdbz_contour, cax=cbar_ax1)
cbar1.set_label("dBZ", fontsize=12)
cbar1.ax.tick_params(labelsize=10)
print(ax_dbz.get_position().height)

# Add the colorbar for the second plot
cbar_ax2 = fig.add_axes([ax_dbz.get_position().x1 + 0.01,
                         ax_dbz.get_position().y0,
                         0.02,
                         ax_dbz.get_position().height])
cbar2 = fig.colorbar(lpi_contour, cax=cbar_ax2)
cbar2.set_label("LPI (j/kg)", fontsize=12)
cbar2.ax.tick_params(labelsize=10)

# Remove the filled contours now the we have made the colorbar
mdbz_contour.remove()

# Set the area we want to view on the plot
ax_dbz.set_extent([-76.965996,-75.090013,43.394115,44.273301],crs=crs.PlateCarree())
ax_wspd.set_extent([-76.965996,-75.090013,43.394115,44.273301],crs=crs.PlateCarree())

#Set the x-axis and  y-axis labels
ax_dbz.set_xlabel("Longitude", fontsize=8)
ax_wspd.set_ylabel("Lattitude", fontsize=8)
ax_dbz.set_ylabel("Lattitude", fontsize=8)

# Add a title
#ax_wspd.set_title("Simulated Composite Reflectivity (dBZ) with LPI (J/kg) and LMA flash points", {"fontsize" : 7})
#ax_dbz.set_title("Observed Composite Reflectivity (dBZ)", {"fontsize" : 7})

plt.savefig("LPIandMDBZComparison.ps",format="eps")
plt.savefig("LPIandMDBZComparison.png",format="png")

plt.show()
