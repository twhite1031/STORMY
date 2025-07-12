import pandas as pd
from metpy.units import units
import metpy.calc as mpcalc
from metpy.plots import SkewT
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
wrf_date_time = datetime(2022,11,18,13,50,00) # Match with flash/sounding or anything you'd want. Will autofind WRF files
domain = 2
SIMULATION = 1 # If comparing runs

observed_flash = False # Find observed or simulated flashes
min_events_per_flash = 10 # Minimum number of sources per flash
min_stations = 6 # more stations = more confident it's a good solution
max_chi = 1 # lower chi^2 = more confident it's a good solution

notes = f"Simulated Flash at Time, Matching Location \nAttempt {SIMULATION}|Domain {domain}" # To be added to whitespace in figure

path = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"
radar_data_dir = '/data2/white/DATA/PROJ_LEE/IOP_2/NEXRADLVL2'

# Variable for observational radar
var = "reflectivity" # "reflectivity", "cross_correlation_ratio", "spectrum_width","differential_phase","velocity"

# Radars to get data from, use None when not needed
radars = ["KTYX"]

radar_coords = {
    'KTYX': (43.7550, -75.6847),  # KTYX - Fort Drum, NY
}

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

# /// Observed and Simulated Flash Setup \\\

# Read in Observed Flashes
if observed_flash == True:

    wrffuncs.get_LMA_flash_data(wrf_date_time,600)
    lma_path = '/data2/white/DATA/PROJ_LEE/LMADATA/LYLOUT_{}000_0600.dat.flash.h5'.format(wrf_date_time.strftime('%y%m%d_%H%M')[:-1])
    print(f"Reading LMA data from {lma_path}")
    timeobj = datetime.strptime(lma_path.split('/')[-1], 
                                           "LYLOUT_%y%m%d_%H%M%S_0600.dat.flash.h5")
    # This is the flash table
    flashes = pd.read_hdf(lma_path,'flashes/LMA_{}00_600'.format(
                                            timeobj.strftime('%y%m%d_%H%M')))
    # This is the event (VHF source) table
    flash_events = pd.read_hdf(lma_path,'events/LMA_{}00_600'.format(
                                          timeobj.strftime('%y%m%d_%H%M')))

    # Make a series of datetime objects for each event
    flash_event_time = np.array([datetime(*wrf_date_time.timetuple()[:3])+timedelta(seconds = j) for j in flash_events.time]) # Retrieve day and calculate event time based on seconds from midnight

    # Select all the sources meeting the criteria set above
    selection_event = (flash_event_time>=wrf_date_time-timedelta(seconds=600))&(flash_event_time < wrf_date_time+timedelta(seconds=600))&(flash_events.chi2<=max_chi)&(flash_events.stations>=min_stations)

    # Step 1: Filter events using your selection criteria
    filtered_events = flash_events[selection_event].copy()
    filtered_events["event_time"] = flash_event_time[filtered_events.index]

    # Step 2: Count valid events per flash_id
    counts = filtered_events.groupby("flash_id").size()
    

    # Get flash_ids with enough points
    valid_flash_ids = counts[counts >= min_events_per_flash].index

    # Filter to only those flash_ids
    filtered_events = filtered_events[filtered_events.flash_id.isin(valid_flash_ids)]

    print(f"Number of valid flashes: {filtered_events.flash_id.nunique()}")

    #Get first lat/lon and time per flash for sounding
    first_latlon = (
        filtered_events.groupby("flash_id")[["lat", "lon", "event_time"]]
        .first()
        .reset_index()
    )
    
    print("Times of flash iniations:\n", first_latlon["event_time"])
    # Check if any flashes are present
    if first_latlon.empty:
        raise ValueError("No flashes found after filtering.")

    # Find the closest flash to your preferred time, no need to manually select within file
    time_diffs = abs(first_latlon["event_time"] - wrf_date_time)
    closest_idx = time_diffs.idxmin()
    closest_flash = first_latlon.loc[closest_idx]

    lat_lon = closest_flash[["lat", "lon"]].tolist()

    print(f"Flash initation found at {lat_lon}, using index {closest_idx}")
    
else:
    
    # Use Simulated Flash

    with Dataset(matched_file) as ds:
        flashi = wrf.getvar(ds, "FLSHI", timeidx=matched_timeidx)
        # Collapse the vertical dimension: check if any level has a non-zero value
        flash_mask_2d = np.any(flashi != 0, axis=0)  # Result: (ny, nx)
        
        # Step 2: Get 2D lat/lon grids
        lats, lons = wrf.latlon_coords(flashi)  # shape: (ny, nx)
        
        flash_lats, flash_lons = wrf.to_np(lats)[flash_mask_2d], wrf.to_np(lons)[flash_mask_2d]
        
        lat_lon = [flash_lats, flash_lons]

        print(lat_lon)

# /// SkewT setup \\\

## Open the NetCDF file
wrfin = Dataset(matched_file)

# Convert desired coordinates to WRF gridbox coordinates
x_y = wrf.ll_to_xy(wrfin, lat_lon[0],lat_lon[1])

# Read skewT variables in (Such as simulated flash)
p1 = wrf.getvar(wrfin,"pressure",timeidx=matched_timeidx)
T1 = wrf.getvar(wrfin,"tc",timeidx=matched_timeidx)
Td1 = wrf.getvar(wrfin,"td",timeidx=matched_timeidx)
u1 = wrf.getvar(wrfin,"ua",units='kts',timeidx=matched_timeidx)
v1 = wrf.getvar(wrfin,"va",units='kts',timeidx=matched_timeidx)
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

fig = plt.figure(figsize=(12,6))

skew1 = SkewT(fig=fig)

# Filter out NaNs first
mask = np.isfinite(T) & np.isfinite(Td)

T_vals = T[mask].values
Td_vals = Td[mask].values
p_vals = p[mask].values

for i, (t, td, pres) in enumerate(zip(T_vals, Td_vals, p_vals)):
    if td > t:
        print(f"[{i}] Pressure: {pres:.1f} hPa — Td: {td:.2f}°C > T: {t:.2f}°C")

# Plot the data using normal plotting functions, in this case using
# log scaling in Y, as dictated by the typical meteorological plot
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

timestamp_str = matched_time.strftime("%Y%m%d_%H%M%S")
lat_str = f"{float(lat_lon[0]):.2f}".replace('.', 'p')
lon_str = f"{float(lat_lon[1]):.2f}".replace('.', 'p').replace('-', 'm') # 'm' for minus, 'p' for period



skew1.ax.set_title(f"WRF Simulated skewT at " + str(matched_time))
plt.suptitle(notes)
plt.savefig(savepath+f"skewT_lat{lat_str}_lon{lon_str}_A{SIMULATION}D{domain}_{timestamp_str}.png")

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
        basename = os.path.basename(file_path)
        # Extract the datetime string
        dt_str = basename[4:19]  # '20221118_081021'

        # Convert to datetime object
        dt_obj = datetime.strptime(dt_str, "%Y%m%d_%H%M%S")

        # Now you can format it for a title or filename
        obs_title_date = dt_obj.strftime("%Y-%m-%d %H:%M:%S")

        radar_data = pyart.io.read(file_path)
        # Process radar_data as needed
   
# Read radar files into Py-ART objects
radar_objects = {}  # Store Py-ART radar objects
display_objects = {}  # Store Py-ART display objects

for radar, file_path in radar_files.items():
    if file_path:
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

# /// Reflectivity Figure \\\

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
ax_WRF.plot(lat_lon[1], lat_lon[0], marker='*', color='brown', transform=crs.PlateCarree(),markersize=10,zorder=10)  
ax_obs.plot(lat_lon[1], lat_lon[0], marker='*', color='brown', transform=crs.PlateCarree(),markersize=10,zorder=10)

# Plotting WSR-88D Radars 
for i, radar in enumerate(radars):
    radar_contour = display_objects[radar].plot_ppi_map(var, mask_outside=True, vmin=settings['vmin'], vmax=settings['vmax'], ax=ax_obs,colorbar_flag=False,title_flag=False,add_grid_lines=False,cmap=settings['cmap'],colorbar_label=settings['label'],zorder=radar_zorder_start + i)

for artist in list(ax_obs.artists):  # list() to avoid modifying during iteration
    artist.remove()

# Manually create a mappable with identical settings to make this work correctly and not mess up the sizes of the plots
norm = mcolors.Normalize(vmin=settings['vmin'], vmax=settings['vmax'])
mappable = ScalarMappable(norm=norm, cmap=settings['cmap'])
mappable.set_array([])

# Add the colorbar for the plots with respect to the position of the plots
cbar_ax1 = fig2.add_axes([ax_obs.get_position().x1+.02,ax_obs.get_position().y0,0.015,ax_obs.get_position().height])
cbar1 = fig2.colorbar(mappable, cax=cbar_ax1)
cbar1.set_label(settings['units'], fontsize=12,labelpad=6)
cbar1.ax.tick_params(labelsize=10)


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



ax_WRF.set_title(f"Simulated Composite Reflectivity at {str(matched_time)}")
ax_obs.set_title(f"Closest Observed Scan at {str(obs_title_date)}")


plt.suptitle(notes)
plt.savefig(savepath+f"WRF_NEXRAD_A{SIMULATION}D{domain}{timestamp_str}.png")
plt.show()
