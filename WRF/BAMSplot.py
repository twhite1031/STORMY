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
import STORMY

# --- USER INPUT ---
wrf_date_time = datetime(2022,11,19,00,00,00)
domain = 2

windbarbs = False # Set to True if you want to plot wind barbs

SIMULATION = "1B" # If comparing runs
path = f"/data1/white/WRF_OUTPUTS/PROJ_LEE/IOP_2/ATTEMPT_1B/"
radar_path = "/data1/white/PYTHON_SCRIPTS/PROJ_LEE/KTYX20221119_000109_V06.ar2v"
lma_path = "/data1/white/Downloads/PROJ_LEE/LMADATA/" 
savepath = f""

# --- END USER INPUT ---

# Build/Find the time data for the model runs
time_df = STORMY.build_time_df(path, domain)
obs_time = pd.to_datetime(wrf_date_time)

# Compute absolute time difference between model times and input time
closest_idx = (time_df["time"] - obs_time).abs().argmin()

# Extract the matched row
match = time_df.iloc[closest_idx]

# Unpack matched file info
matched_file = match["filename"]
matched_timeidx = match["timeidx"]
matched_time = match["time"]
print(f"Closest match: {matched_time} in file {matched_file} at time index {matched_timeidx}")

# Get the observed variables
radar_data = obs_dbz = pyart.io.read_nexrad_archive(radar_path)
display = pyart.graph.RadarMapDisplay(obs_dbz)

# Open the LMA NetCDF file
lmafilename = "LYLOUT_221119_001000_0600.dat.gz"
ds, starttime  = lma_read.dataset(lmapath + lmafilename)

# Read in data from matched WRF file
with Dataset(matched_file) as ds:
    mdbz = getvar(ds, "mdbz",timeidx=matched_timeidx)
    lpi = getvar(ds, "LPI",timeidx=matched_timeidx)


# Get the lat/lon points and projection object from WRF data
lats, lons = latlon_coords(cape)
cart_proj = get_cartopy(cape)
WRF_ylim = cartopy_ylim(cape)
WRF_xlim = cartopy_xlim(cape)

# Create a figure that will have 3 subplots
fig = plt.figure(figsize=(30,15))
ax_WRF = fig.add_subplot(1,2,1, projection=cart_proj)
ax_obs = fig.add_subplot(1,2,2, projection=cart_proj)
axs = [ax_WRF, ax_obs]

# Read in high definition counties
reader = shpreader.Reader('countyl010g.shp')
counties = list(reader.geometries())
COUNTIES = cfeature.ShapelyFeature(counties, crs.PlateCarree(),zorder=5)

# Viewing area of the plot
extent = [-76.965996,-75.090013,43.394115,44.273301],

for ax in as:
    ax.margins(x=0,y=0,tight=True)
    ax.add_feature(COUNTIES, facecolor='none', edgecolor='black',linewidth=1)
    ax.set_xlim(WRF_xlim)
    ax.set_ylim(WRF_ylim)
    ax.set_extent(extent,crs=crs.PlateCarree())
    STORMY.add_cartopy_features(ax)

# Define value boundaries and colormap
levels = np.arange(0, 75, 5)  # 0 to 70 dBZ in 5 dBZ bins
cmap = plt.get_cmap("NWSRef")  # or your defined reflectivity colormap

# Create a BoundaryNorm for shared normalization
norm = BoundaryNorm(boundaries=levels, ncolors=cmap.N, clip=True)

# Plot contour lines of simulated maximum reflectivity, with filled contours used to get the colorbar
mdbz_contourline = ax_WRF.contour(to_np(lons), to_np(lats),mdbz,cmap=cmap, norm=norm,transform=crs.PlateCarree(),zorder=2)
mdbz_contour = ax_WRF.contourf(to_np(lons), to_np(lats), mdbz,cmap=cmap,norm=norm, transform=crs.PlateCarree(),zorder=0)

# Add LPI to the plot with simulated reflectivity
lpi_levels = np.arange(0,6,1)
lpi_contour = ax_WRF.contourf(to_np(lons), to_np(lats),to_np(lpi),levels=lpi_levels, transform=crs.PlateCarree(),cmap=cmap)

# Calculate composite reflectivity from level II data
comp_ref = pyart.retrieve.composite_reflectivity(obs_dbz, field="reflectivity")
display = pyart.graph.RadarMapDisplay(comp_ref)

obs_contour = display.plot_ppi_map("composite_reflectivity",vmin=0,vmax=75,mask_outside=True,ax=ax_obs, colorbar_flag=False, title_flag=False, add_grid_lines=False, cmap=nwscmap,zorder=2)

# Filter LMA points
#count_subset = {'number_of_events':slice(0,10,1)}

station_filter = (ds.event_stations >= 7)
chi_filter = (ds.event_chi2 <= 1.0)

filter_subset = {'number_of_events':(chi_filter & station_filter)}

# Use this line to filter number of events by slices
#filter1 = ds[filter_subset].isel({'number_of_events':slice(0,10)})

# note that we first filter on all the data select 10000 points, and then on that dataset we further filter 
art = ds[filter_subset].plot.scatter(x='event_longitude', y='event_latitude',c='black', ax=ax_WRF, 
                    s=10, vmin=0.0,edgecolor='black',vmax=5000,transform=crs.PlateCarree())

# KART flash observation
ax_WRF.scatter([-76.023498],[43.990428],c='red',s=75,marker="X",transform=crs.PlateCarree())

## Create a color bar for both plots
fig.subplots_adjust(right=0.8)

# Add the colorbar for the first plot
cbar_ax1 = fig.add_axes([ax_obs.get_position().x1 + 0.07,
                         ax_obs.get_position().y0,
                         0.02,
                         ax_obs.get_position().height])

cbar1 = fig.colorbar(mdbz_contour, cax=cbar_ax1)
cbar1.set_label("dBZ", fontsize=12)
cbar1.ax.tick_params(labelsize=10)

# Add the colorbar for the second plot
cbar_ax2 = fig.add_axes([ax_obs.get_position().x1 + 0.01,
                         ax_obs.get_position().y0,
                         0.02,
                         ax_obs.get_position().height])
cbar2 = fig.colorbar(lpi_contour, cax=cbar_ax2)
cbar2.set_label("LPI (j/kg)", fontsize=12)
cbar2.ax.tick_params(labelsize=10)

# Remove the filled contours now the we have made the colorbar
mdbz_contour.remove()

#Set the x-axis and  y-axis labels
ax_obs.set_xlabel("Longitude", fontsize=8)
ax_WRF.set_ylabel("Lattitude", fontsize=8)
ax_obs.set_ylabel("Lattitude", fontsize=8)

# Add a title
#ax_WRF.set_title("Simulated Composite Reflectivity (dBZ) with LPI (J/kg) and LMA flash points", {"fontsize" : 7})
#ax_obs.set_title("Observed Composite Reflectivity (dBZ)", {"fontsize" : 7})

plt.savefig("BAMSplot.png",format="png")
plt.show()
