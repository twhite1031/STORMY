import pandas as pd
from metpy.units import units
import metpy.calc as mpcalc
from metpy.plots import SkewT, ctables
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import wrf
from netCDF4 import Dataset
import numpy as np
from datetime import datetime, timedelta
import os 
import wrffuncs
import radarfuncs
import pyart
import nexradaws
conn = nexradaws.NexradAwsInterface()

# --- USER INPUT ---
wrf_date_time = datetime(2022,11,19,00,00,00)
domain = 2
SIMULATION = 1 # If comparing runs

notes = f"IOP2 Oswego Sounding Comparison\nAttempt {SIMULATION}\nDomain {domain}" # To be added to whitespace in figure

path = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"
sounding_path = "/data2/white/DATA/PROJ_LEE/IOP_2/SOUNDINGS/edt_20221118_2357.txt" 
radar_data_dir = 'C:/Users/thoma/Documents/DATA/LESPaRC/NEXRADLVL2'

# Variable for observational radar
var = "reflectivity" # "reflectivity", "cross_correlation_ratio", "spectrum_width","differential_phase","velocity"

# Radars to get data from, use None when not needed
radars = ["KTYX"]

radar_coords = {
    'KTYX': (43.7550, -75.6847),  # KTYX - Fort Drum, NY
}

lat_lon = [43.4509, -76.5434] # Coordinates for Simulated Sounding

savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"

# --- END USER INPUT ---

# Build/Find the time data for the model runs
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

# Define the column names based on the data structure
column_names = ['Time', 'Height', 'Pressure', 'Temperature', 'Relative_Humidity', 'Wind_Direction', 'Wind_Speed', 'Range', 'Latitude', 'Longitude']

df = pd.read_csv(sounding_path, delim_whitespace=True, header=None, names=column_names)

# Convert the necessary columns to appropriate units
p  = df['Pressure'].values * units.hPa
temp  = df['Temperature'].values * units.degC
wspd = df['Wind_Speed'].values * units.meter / units.second
wdir = df['Wind_Direction'].values * units.deg
rh = df['Relative_Humidity'].values * units.percent
dewtemp = mpcalc.dewpoint_from_relative_humidity(temp, rh)

u_v = mpcalc.wind_components(wspd, wdir)
u = u_v[0]
v = u_v[1]

# Create a figure with two subplots
fig = plt.figure(figsize=(12, 6))

# Example of defining your own vertical barb spacing
skew = SkewT(fig=fig, subplot=(1,2,1))

# Plot the data using normal plotting functions, in this case using
# log scaling in Y, as dictated by the typical meteorological plot
skew.plot(p, temp, 'r')
skew.plot(p, dewtemp, 'g')

# Plot only values nearest to defined interval values
skew.plot_barbs(p[::50], u[::50], v[::50])

# Add the relevant special lines
skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines()
skew.ax.set_ylim(1000, 100)
skew.ax.set_xlim(-40, 20)
skew.ax.set_xlabel('Temperature ($^\circ$C)')
skew.ax.set_ylabel('Pressure (hPa)')

wrfin = Dataset(matched_file)

# Convert desired coorindates to WRF gridbox coordinates
x_y = wrf.ll_to_xy(wrfin, lat_lon[0], lat_lon[1])

# Read skewT variables in
p1 = wrf.getvar(wrfin,"pressure",timeidx=matched_timeidx)
T1 = wrf.getvar(wrfin,"tc",timeidx=matched_timeidx)
Td1 = wrf.getvar(wrfin,"td",timeidx=matched_timeidx)
u1 = wrf.getvar(wrfin,"ua",timeidx=matched_timeidx)
v1 = wrf.getvar(wrfin,"va",timeidx=matched_timeidx)
mdbz = wrf.getvar(wrfin, "mdbz", timeidx=matched_timeidx)

# Get variables for desired coordinates
p = p1[:,x_y[1],x_y[0]] * units.hPa
T = T1[:,x_y[1],x_y[0]] * units.degC
Td = Td1[:,x_y[1],x_y[0]] * units.degC
u = u1[:,x_y[1],x_y[0]] * units('kt')
v = v1[:,x_y[1],x_y[0]] * units('kt')

#Test if the coordinates are correct
lat1 = wrf.getvar(wrfin,"lat",timeidx=matched_timeidx)
lon1 = wrf.getvar(wrfin,"lon",timeidx=matched_timeidx)
lat = lat1[x_y[1],x_y[0]] * units.degree_north
lon = lon1[x_y[1],x_y[0]] * units.degree_east

print(f"Viewing the lattitude and longitude point of {lat.values}, {lon.values}")

# ///Radar Logic\\\

radar_files = {}

for radar in radars:
    if radar is not None:
        file_path = radarfuncs.find_closest_radar_file(matched_time, radar_data_dir, radar)

        # Check if the file exists before storing it
        if file_path and os.path.exists(file_path):
            radar_files[radar] = file_path
        else:
            radar_files[radar] = None  # Mark as missing

# Check for missing files and handle downloading if needed
missing_radars = [radar for radar, file in radar_files.items() if file is None]


if missing_radars:
    print(f"Missing radar files for: {missing_radars}")

    # Define the time range for available scans
    start = matched_time - timedelta(minutes=3)
    end = matched_time + timedelta(minutes=3)

    # Check available scans and download data for missing radars
    for radar in missing_radars:
        scans = conn.get_avail_scans_in_range(start, end, radar)
        print(f"There are {len(scans)} scans available for {radar} between {start} and {end}\n")

        # Download the data
        downloaded_files = conn.download(scans, radar_data_dir)

    # **Step 3: Re-check for Files After Download**
for radar in missing_radars:
    file_path = radarfuncs.find_closest_radar_file(matched_time, radar_data_dir, radar)
    if file_path and os.path.exists(file_path):
        radar_files[radar] = file_path
    else:
        print(f"Warning: Still missing file for {radar} after download.")

# **Step 4: Read Radar Data**
for radar, file_path in radar_files.items():
    if file_path:
        print(f"Reading radar data from {file_path} for {radar}...")
        radar_data = pyart.io.read(file_path)
        # Process radar_data as needed
   
print(radar_files)

# Read radar files into Py-ART objects
radar_objects = {}  # Store Py-ART radar objects
display_objects = {}  # Store Py-ART display objects

for radar, file_path in radar_files.items():
    if file_path:
        print(f"Reading radar data from {file_path} for {radar}...")
        radar_obj = pyart.io.read(file_path)
        radar_objects[radar] = radar_obj  # Store radar object
        display_objects[radar] = pyart.graph.RadarMapDisplay(radar_obj)  # Create display object

# ZDR cmap
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
    # Add more fields as needed
}

settings = field_settings[var]
radar_zorder_start = 2

# Example of defining your own vertical barb spacing
skew1 = SkewT(fig=fig,subplot=(1,2,2))

# Plot the data using normal plotting functions, in this case using
# log scaling in Y, as dictated by the t<F9>ypical meteorological plot
skew1.plot(p, T, 'r')
skew1.plot(p, Td, 'g')

# Set spacing interval--Every 50 mb from 1000 to 100 mb
my_interval = np.arange(100, 1000, 50) * units('mbar')

# Get indexes of values closest to defined interval
ix = mpcalc.resample_nn_1d(p, my_interval)

# Plot only values nearest to defined interval values
skew1.plot_barbs(p[ix], u[ix], v[ix])

# Add the relevant special lines
skew1.plot_dry_adiabats()
skew1.plot_moist_adiabats()
skew1.plot_mixing_lines()
skew1.ax.set_ylim(1000, 100)
skew1.ax.set_xlim(-40, 20)
skew1.ax.set_xlabel('Temperature ($^\circ$C)')
skew1.ax.set_ylabel('Pressure (hPa)')

file_date, title_date = wrffuncs.parse_wrfout_time(os.path.basename(matched_file)) # Get date of WRF file for title and file naming

skew.ax.set_title(f"Observed skewT")
skew1.ax.set_title(f"WRF Simulated skewT at " + title_date)

plt.savefig(savepath+f"OSW_skewTA{SIMULATION}D{domain}{file_date}.png")
plt.suptitle(notes)

# Get the latitude and longitude points
lats, lons = wrf.latlon_coords(mdbz)

# Get the cartopy mapping object
cart_proj = wrf.get_cartopy(mdbz)

# Create a figure that will have 2 subplots
fig2 = plt.figure(figsize=(30,15))
ax_WRF = fig2.add_subplot(1,2,1, projection=cart_proj)
ax_obs = fig2.add_subplot(1,2,2, projection=cart_proj)

ax_WRF.set_xlim(wrf.cartopy_xlim(mdbz))
ax_WRF.set_ylim(wrf.cartopy_ylim(mdbz))

ax_obs.set_xlim(wrf.cartopy_xlim(mdbz))
ax_obs.set_ylim(wrf.cartopy_ylim(mdbz))


# Download and add the states, lakes and coastlines
states = NaturalEarthFeature(category="cultural", scale="50m", facecolor="none", name="admin_1_states_provinces")
ax_WRF.add_feature(states, linewidth=.1, edgecolor="black")
ax_WRF.add_feature(cfeature.LAKES.with_scale('50m'),linewidth=1, facecolor="none",  edgecolor="black")

levels = np.arange(5, 71, 5)

# Make the filled countours with specified levels and range
qcs = ax_WRF.contourf(wrf.to_np(lons), wrf.to_np(lats),mdbz,levels=levels,transform=crs.PlateCarree(),cmap="NWSRef")
ax_WRF.plot(lat_lon[1], lat_lon[0], marker='*', color='brown', transform=crs.PlateCarree(),markersize=10,zorder=10)  # 'ro' means red color ('r') and circle marker ('o')
ax_obs.plot(lat_lon[1], lat_lon[0], marker='*', color='brown', transform=crs.PlateCarree(),markersize=10,zorder=10)  # 'ro' means red color ('r') and circle marker ('o')

# Manually create a mappable with identical settings to make this work correctly and not mess up the sizes of the plots
norm = mcolors.Normalize(vmin=settings['vmin'], vmax=settings['vmax'])
mappable = ScalarMappable(norm=norm, cmap=settings['cmap'])
mappable.set_array([])

# Add the colorbar for the plots with respect to the position of the plots
cbar_ax1 = fig2.add_axes([ax_obs.get_position().x1 + 0.02,ax_obs.get_position().y0,0.015,ax_obs.get_position().height])
cbar1 = fig2.colorbar(mappable, cax=cbar_ax1)
cbar1.set_label(settings['units'], fontsize=12,labelpad=6)
cbar1.ax.tick_params(labelsize=10)

# Plotting WSR-88D Radars 
for i, radar in enumerate(radars):
    radar_contour = display_objects[radar].plot_ppi_map(var, mask_outside=True, vmin=settings['vmin'], vmax=settings['vmax'], ax=ax_obs,colorbar_flag=False,title_flag=False,add_grid_lines=False,cmap=settings['cmap'],colorbar_label=settings['label'],zorder=radar_zorder_start + i)

for artist in list(ax_obs.artists):  # list() to avoid modifying during iteration
    artist.remove()

ax_obs.add_feature(states, linewidth=.1, edgecolor="black")
ax_obs.add_feature(cfeature.LAKES.with_scale('50m'),linewidth=1, facecolor="none",  edgecolor="black")

# Add the gridlines
gl_WRF = ax_WRF.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=False)
gl_WRF.xlabel_style = {'rotation': 'horizontal','size': 14,'ha':'center'} # Change 14 to your desired font size
gl_WRF.ylabel_style = {'size': 14}  # Change 14 to your desired font size
gl_WRF.xlines = True
gl_WRF.ylines = True

gl_WRF.top_labels = False  # Disable top labels
gl_WRF.right_labels = False  # Disable right labels
gl_WRF.xpadding = 20  

gl_obs = ax_obs.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=False)
gl_obs.xlabel_style = {'rotation': 'horizontal','size': 14,'ha':'center'} # Change 14 to your desired font size
gl_obs.ylabel_style = {'size': 14}  # Change 14 to your desired font size
gl_obs.xlines = True
gl_obs.ylines = True

gl_obs.top_labels = False  # Disable top labels
gl_obs.right_labels = False  # Disable right labels
gl_obs.xpadding = 20  



ax_WRF.set_title(f"Simulated Composite Reflectivity at {title_date}")
ax_obs.set_title(f"Neareast Observed Scan")


plt.suptitle(notes)
plt.savefig(savepath+f"WRF_NEXRAD_A{SIMULATION}D{domain}{file_date}.png")
plt.show()
