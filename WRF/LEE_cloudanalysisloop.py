import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import from_levels_and_colors
import matplotlib.patches as patches
from cartopy import crs
from netCDF4 import Dataset
from wrf import (getvar, to_np, get_cartopy, latlon_coords, vertcross,
                 interpline, interplevel, CoordPair, xy_to_ll,ll_to_xy,cartopy_xlim, cartopy_ylim)
from scipy.ndimage import label,  generate_binary_structure
from skimage.measure import regionprops
import cartopy.feature as cfeature
import wrffuncs
from datetime import datetime
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
import sys
import os
import pyart #For cmap
import xarray as xr
"""
A script used to automatically identify Lake-effect band locations based on connected cloud fractions at a certain level, 
then classifying the region as a cloud band for analysis
"""

# --- USER INPUT ---
start_time, end_time  = datetime(2022,11,18,13,30), datetime(2022, 11, 18,14, 00)
domain = 2

INTERACTIVE = False # Set to False for non-interactive (auto-run) mode
SHOW_FIGS = False # Do not display completed figures
USE_MAX_DBZ = True  # Toggle this to switch between lat/lon and max dBZ

# Threshold to identify the snow band (e.g., cloud fraction > .1)
threshold = 1
lat_lon = [43.86935, -76.164764]  # Coordinates to start cloud check
ht_level = 15

aspect_ratio_thresh = 2.5

# Define thresholds for LLAP band size (in gridpoints)
min_gridboxes = 100   # minimum connected gridboxes
max_gridboxes = 1000 # maximum connected gridboxes

SIMULATION = 1 # If comparing runs
path = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"
savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"

# --- END USER INPUT ---

# Build/Find the time data for the model runs
time_df = wrffuncs.build_time_df(path, domain)

# Filter time range based on start_time and end_time
time_mask = (time_df["time"] >= start_time) & (time_df["time"] <= end_time)

time_df = time_df[time_mask].reset_index(drop=True)

filelist = time_df["filename"].tolist()
timelist = time_df["time"].tolist()
timeidxlist = time_df["timeidx"].tolist()


# Define dict of variables we will store over time
cloud_with_flash = {
    "height_agl": [],
    "reflectivity": [],
    "SNOW": [],
    "GRAUPEL": [],
    "ICE": [],
    "WATER_VAPOR": [],
    "CLOUD": [],
    "HAIL": []
}
cloud_without_flash = {
    "height_agl": [],
    "reflectivity": [],
    "SNOW": [],
    "GRAUPEL": [],
    "ICE": [],
    "WATER_VAPOR": [],
    "CLOUD": [],
    "HAIL": []
}

cloud_all = {
    "height_agl": [],
    "reflectivity": [],
    "SNOW": [],
    "GRAUPEL": [],
    "ICE": [],
    "WATER_VAPOR": [],
    "CLOUD": [],
    "HAIL": []
}

cloud_all_with_lightning = {
    "height_agl": [],
    "reflectivity": [],
    "SNOW": [],
    "GRAUPEL": [],
    "ICE": [],
    "WATER_VAPOR": [],
    "CLOUD": [],
    "HAIL": []
}

cloud_all_without_lightning = {
    "height_agl": [],
    "reflectivity": [],
    "SNOW": [],
    "GRAUPEL": [],
    "ICE": [],
    "WATER_VAPOR": [],
    "CLOUD": [],
    "HAIL": []
}

# !!! CREATE FUNCTIONS !!!

# === Prompt function for plotting ===
def prompt_plot(prompt_text, plot_func):

    valid_responses = {'y', 'n', 'exit'}

    while True:

        if not INTERACTIVE:
            print(f"[AUTO] {prompt_text} ... running plot.")
            plot_func()

            break

        response = input(f"{prompt_text} (y/n/exit): ").strip().lower()
    
        if response not in valid_responses:
            print("Invalid input. Please type 'y' for yes, 'n' for no, or 'exit' to quit.")
            continue
    
        if response == 'y':
            plot_func()
            break
        elif response == 'n':
            print("Skipped.\n")
            break
        else:  
            print("Exiting script.")
            sys.exit()

# === Scrolling function for finished plots ===
def show_fig_with_keypress(key='enter'):
    fig = plt.gcf()
    ax = plt.gca()
    pressed = plt.waitforbuttonpress()
    plt.close()

# === Plot simulated maximum reflectivity ===
def plot_mdbz(max_dbz,mask_cases):
    
    # Collapse cloud mask (this one should always exist)
    cloud_mask_2d = np.max(mask_cases["cloud"], axis=0)
        
    # Get the lat/lon points 
    lats, lons = latlon_coords(max_dbz)

    # Get the cartopy projection object
    cart_proj = get_cartopy(max_dbz)

    # Create the figure
    fig = plt.figure(figsize=(12,9))
    ax = plt.axes(projection=cart_proj)

    # Download and create the states, land, and oceans using cartopy features
    states = cfeature.NaturalEarthFeature(category='cultural', scale='50m',
                                      facecolor='none',
                                      name='admin_1_states_provinces')
    land = cfeature.NaturalEarthFeature(category='physical', name='land',
                                    scale='50m',
                                    facecolor=cfeature.COLORS['land'])

    lakes = cfeature.NaturalEarthFeature(category='physical',name='lakes',scale='50m',facecolor="none",edgecolor="blue")
    ocean = cfeature.NaturalEarthFeature(category='physical', name='ocean',
                                     scale='50m',
                                    facecolor=cfeature.COLORS['water'])

    # Add Cartopy features
    ax.add_feature(land,zorder=0)
    ax.add_feature(ocean,zorder=0)
    ax.add_feature(lakes,zorder=0)
    ax.add_feature(states, edgecolor='gray',zorder=2)
    
    # Add the gridlines
    gl = ax.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=False)
    gl.xlabel_style = {'rotation': 'horizontal','size': 14,'ha':'center'} # Change 14 to your desired font size
    gl.ylabel_style = {'size': 14}  # Change 14 to your desired font size
    gl.xlines = True
    gl.ylines = True

    gl.top_labels = False  # Disable top labels
    gl.right_labels = False  # Disable right labels
    gl.xpadding = 20

    # Plot Reflectivity Contours
    levels = [5,10, 15, 20, 25, 30, 35, 40, 45,50,55,60]
    cf = ax.contourf(to_np(lons), to_np(lats), to_np(max_dbz), levels=levels,cmap="NWSRef",transform=crs.PlateCarree(), zorder=4)

    # Final touches
    ax.set_xlim(cartopy_xlim(max_dbz))
    ax.set_ylim(cartopy_ylim(max_dbz))
    plt.title(f"Simulated Composite Reflectivity at {matched_time}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.suptitle(
        f"Mean height {mean_start_height} index {ht_level} | {len(lat_inds)} gridboxes | Threshold {threshold} | Attempt {SIMULATION}\n"
        f"Starting Check at {lat_lon[0]} {lat_lon[1]}"
    )

    # Save the figure
    os.makedirs(savepath, exist_ok=True)
    filename = f"plancloud_ht{ht_level}T{threshold}_A{SIMULATION}D{domain}_{timestamp_str}.png"
    plt.savefig(os.path.join(savepath, filename), dpi=150)
    if SHOW_FIGS:
        plt.show(block=False)
        print("Press any key to continue...")
        plt.waitforbuttonpress()
        plt.close()
    else:
        plt.close()

# === Plan view of cloud and flash regions ===
def plot_plan_cloud(max_dbz, mask_cases):
    '''
    Plots a 2D plan-view map showing cloud regions and flash-associated cloud regions.

    Parameters:
    -----------
    max_dbz : xarray.DataArray 
        Reflectivity variable used to extract latitude, longitude, and map projection info.
        Not directly used for plotting, but essential for geolocation.

    mask_cases : dict
        Dictionary containing boolean or integer cloud masks with keys:
        - "cloud" (3D array): Binary mask of cloud regions (1 where cloud exists, 0 elsewhere).
        - "cloudlight" (optional, 3D array): Binary mask of cloud regions associated with lightning.

    Notes:
    ------
    - Requires several global variables to be defined in the script:
        `matched_time`, `mean_start_height`, `ht_level`, `lat_inds`,
        `threshold`, `SIMULATION`, `lat_lon`, `savepath`, `domain`,
        `timestamp_str`, `SHOW_FIGS`.
    - Saves the figure to disk and optionally displays it.
    '''
    # Collapse cloud mask (this one should always exist)
    cloud_mask_2d = np.max(mask_cases["cloud"], axis=0)

    # Optional masks: flash_mask and no_flash_mask
    flash_mask_2d = np.max(mask_cases["cloudlight"], axis=0) if "cloudlight" in mask_cases else None
        
    # Get the lat/lon points 
    lats, lons = latlon_coords(max_dbz)

    # Get the cartopy projection object
    cart_proj = get_cartopy(max_dbz)

    # Create the figure
    fig = plt.figure(figsize=(12,9))
    ax = plt.axes(projection=cart_proj)

    # Download and create the states, land, and oceans using cartopy features
    states = cfeature.NaturalEarthFeature(category='cultural', scale='50m',
                                      facecolor='none',
                                      name='admin_1_states_provinces')

    land = cfeature.NaturalEarthFeature(category='physical', name='land',
                                    scale='50m',
                                    facecolor=cfeature.COLORS['land'])

    lakes = cfeature.NaturalEarthFeature(category='physical',name='lakes',scale='50m',facecolor="none",edgecolor="blue")
    ocean = cfeature.NaturalEarthFeature(category='physical', name='ocean',
                                     scale='50m',
                                    facecolor=cfeature.COLORS['water'])

    # Add Cartopy features
    ax.add_feature(land,zorder=0)
    ax.add_feature(ocean,zorder=0)
    ax.add_feature(lakes,zorder=0)
    ax.add_feature(states, edgecolor='gray',zorder=2)
    
    # Add the gridlines
    gl = ax.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=False)
    gl.xlabel_style = {'rotation': 'horizontal','size': 14,'ha':'center'} # Change 14 to your desired font size
    gl.ylabel_style = {'size': 14}  # Change 14 to your desired font size
    gl.xlines = True
    gl.ylines = True

    gl.top_labels = False  # Disable top labels
    gl.right_labels = False  # Disable right labels
    gl.xpadding = 20

    # Plot Cloud Region (red)
    cf = ax.contourf(
        to_np(lons), to_np(lats), to_np(cloud_mask_2d.astype(float)),
        levels=[0.5, 1.5], colors=['red'], alpha=0.3,
        transform=crs.PlateCarree(), zorder=4
    )

    # Plot Flash Region (yellow)
    if flash_mask_2d is not None:
        cf_flash = ax.contourf(
            to_np(lons), to_np(lats), to_np(flash_mask_2d.astype(float)),
            levels=[0.5, 1.5], colors=['yellow'], alpha=0.6,
            transform=crs.PlateCarree(), zorder=5
        )

     
    # Add patch legend
    legend_elements = [
        Patch(facecolor='red', edgecolor='red', label='Cloud Region'),
        Patch(facecolor='yellow', edgecolor='yellow', label='Flash Region')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Final touches
    ax.set_xlim(cartopy_xlim(max_dbz))
    ax.set_ylim(cartopy_ylim(max_dbz))
    plt.title(f"Cloud region linked to flash at {matched_time}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.suptitle(
        f"Mean height {mean_start_height} index {ht_level} | {len(lat_inds)} gridboxes | Threshold {threshold} | Attempt {SIMULATION}\n"
        f"Starting Check at {lat_lon[0]} {lat_lon[1]}"
    )

    # Save the figure
    os.makedirs(savepath, exist_ok=True)
    filename = f"plancloud_ht{ht_level}T{threshold}_A{SIMULATION}D{domain}_{timestamp_str}.png"
    plt.savefig(os.path.join(savepath, filename), dpi=150)
    if SHOW_FIGS:
        plt.show(block=False)
        print("Press any key to continue...")
        plt.waitforbuttonpress()
        plt.close()
    else:
        plt.close()

def plot_q_histogram(var, mask):
    '''
    Plots a 1D histogram of mixing ratios (g/kg) for data within a masked region.

    Parameters:
    -----------
    var : np.ndarray or xarray.DataArray
        3D field of specific humidity or mixing ratio values in g/kg.

    mask : np.ndarray of bool
        Boolean array with the same shape as `var` indicating where to include values.

    Notes:
    ------
    Uses global variables: `name`, `matched_time`, `mean_start_height`, `ht_level`,
    `lat_inds`, `threshold`, `SIMULATION`, `domain`, `timestamp_str`, `savepath`,
    `SHOW_FIGS`, and `case_name`.
    '''
    flat_data = var[mask]
    flat_data = flat_data[~np.isnan(flat_data)]

    plt.figure(figsize=(8, 10))
    plt.hist(flat_data, bins=50, edgecolor='black')
    plt.title(f"{name} Mixing Ratio Histogram (g/kg) at {matched_time}")
    plt.xlabel("Mixing Ratio (g/kg)")
    plt.ylabel("Frequency")
    plt.grid(True)

    plt.suptitle(
        f"Mean height {mean_start_height:.1f}m | Level {ht_level} | Total Gridboxes: {len(lat_inds)} | "
        f"T: {threshold} | A{SIMULATION} D{domain}"
    )

    filename = f"{case_name}_hist{name[:3]}_ht{ht_level}T{threshold}_A{SIMULATION}D{domain}_{timestamp_str}.png"
    plt.savefig(os.path.join(savepath, filename))
    plt.show() if SHOW_FIGS else plt.close()

def plot_q_2d_histogram(var, mask):
    '''
    Plots a 2D histogram of mixing ratio frequency versus height for a masked region.

    Parameters:
    -----------
    var : np.ndarray or xarray.DataArray
        3D field of specific humidity or mixing ratio values in g/kg.

    mask : np.ndarray of bool
        Boolean mask with same shape as `var`, indicating where to sample data.

    Notes:
    ------
    Assumes vertical dimension is the first axis (shape: [z, y, x]).
    Uses global variables: `name`, `ht_agl`, `matched_time`, `threshold`, `ht_level`,
    `SIMULATION`, `domain`, `timestamp_str`, `savepath`, `SHOW_FIGS`, and `case_name`.
    '''
    flat_data = var[mask]
    flat_data = flat_data[~np.isnan(flat_data)]

    z_levels = var.shape[0]
    mix_bins = np.linspace(0, np.nanmax(flat_data), 100)
    masked_var = np.where(mask, var, np.nan)

    mean_heights = []
    hist_rows = []
    
    for z in range(z_levels):
        if not np.any(mask[z]):
            continue
        level_data = masked_var[z].flatten()
        level_data = level_data[~np.isnan(level_data)]
        if len(level_data) == 0:
            continue
        hist, _ = np.histogram(level_data, bins=mix_bins)
        hist_rows.append(hist)

        level_heights = ht_agl[z][mask[z]]
        mean_heights.append(np.nanmean(level_heights))

    hist_2d = np.array(hist_rows)
    mean_heights = np.array(mean_heights)

    plt.figure(figsize=(10, 8))
    plt.pcolormesh(mix_bins[:-1], mean_heights, hist_2d, shading='auto')
    plt.xlabel("Mixing Ratio (g/kg)")
    plt.ylabel("Mean Cloud Height (m)")
    plt.title(f"{name} 2D Histogram ({case_name}) at {matched_time}")
    plt.colorbar(label="Frequency")
    plt.tight_layout()

    filename2d = f"{case_name}_2Dhist{name[:3]}_ht{ht_level}T{threshold}_A{SIMULATION}D{domain}_{timestamp_str}.png"
    plt.savefig(os.path.join(savepath, filename2d))
    plt.show() if SHOW_FIGS else plt.close()

def plot_w_histogram(vert_velocity, mask):
    '''
    Plots a histogram of vertical velocity (m/s) values within a masked region.

    Parameters:
    -----------
    mask : np.ndarray of bool
        Boolean array to apply to `vert_velocity` for selecting valid points.

    Notes:
    ------
    Uses global variable `vert_velocity` for the vertical velocity field.
    Also depends on `matched_time`, `mean_start_height`, `ht_level`, `lat_inds`,
    `threshold`, `SIMULATION`, `domain`, `timestamp_str`, `savepath`, `SHOW_FIGS`,
    and `case_name`.
    '''
    flat_data = vert_velocity[mask]
    flat_data = flat_data[~np.isnan(flat_data)]

    plt.figure(figsize=(8, 10))
    plt.hist(flat_data, bins=50, edgecolor='black')
    plt.title(f"Vertical Velocity Histogram (m/s) at {matched_time}")
    plt.xlabel("Vertical Velocity (m/s)")
    plt.ylabel("Frequency")
    plt.grid(True)

    plt.suptitle(
        f"Mean height {mean_start_height:.1f}m | Level {ht_level} | Gridboxes: {len(lat_inds)} | "
        f"T: {threshold} | A{SIMULATION} D{domain}"
    )

    filename = f"{case_name}_histw_ht{ht_level}T{threshold}_A{SIMULATION}D{domain}_{timestamp_str}.png"
    plt.savefig(os.path.join(savepath, filename))
    plt.show() if SHOW_FIGS else plt.close()

# === 3D Voxel Plot (grid-based, no lat/lon) === 
def plot_3d_voxel(vert_velocity, land_mask, mask_cases):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # === Expand land mask to shape (1, y, x) and convert to NumPy ===
    land_voxel_layer = to_np(land_mask.expand_dims(bottom_top=[0]))  # (1, y, x)

    # === Extract cloud mask at base level (y, x) and cast to boolean ===
    cloud_start = to_np(mask_cases["cloud"][ht_level, :, :]).astype(bool)  # (y, x)

    # === Extract 2D slice of land layer ===
    land_2d = land_voxel_layer[0]  # shape: (y, x)

    # === Mask land/water values to only show where cloud exists ===
    land_masked = np.where(cloud_start, land_2d, np.nan)  # (y, x)

    # === Add back z=0 dimension to make (1, y, x)
    land_masked_3d = land_masked[np.newaxis, :, :]  # (1, y, x)

    # === Transpose to (x, y, z) for voxel plotting
    land_voxels = np.transpose(land_masked_3d, (2, 1, 0))  # shape: (x, y, z=1)

    # === Define colors
    land_colors = np.empty(land_voxels.shape, dtype=object)
    land_colors[land_voxels == 1] = 'tan'   # land
    land_colors[land_voxels == 0] = 'blue'  # water

    # === Plot: mask out np.nan from display
    facecolors = np.ma.masked_where(np.isnan(land_voxels), land_colors)
    mask = ~np.isnan(land_voxels)

    ax.voxels(mask, facecolors=facecolors, alpha=0.6)
    # Print original shape (z, y, x)
    z_dim, y_dim, x_dim = mask_cases["cloud"].shape
    print("Original cloud mask shape (z, y, x):", z_dim, y_dim, x_dim)

    # Transpose to match (x, y, z) voxel format
    cloud_mask_vox = np.transpose(mask_cases["cloud"], (2, 1, 0)) # Downsample here if needed (e.g., [::2, ::2, ::2])
    print("Voxel shape (x, y, z):", cloud_mask_vox.shape)

    # *** New Updraft Check ***
    updraft_threshold = 2.0 # m/s
    strong_updraft_mask = np.transpose((mask_cases["cloud"]) & (vert_velocity > updraft_threshold), (2, 1, 0))

    # Plot voxels
    ax.voxels(strong_updraft_mask, facecolors='magenta',alpha=0.6,label="Cloud")
    ax.voxels(cloud_mask_vox, facecolors='lightblue', edgecolor=None, alpha=0.4,label="Vertical Velocity > 2 m/s")
   
    # Axes and appearance
    ax.view_init(elev=10, azim=-90)

    # Psuedo lat/lon ticks
    lat2d, lon2d = latlon_coords(max_dbz)

    # Sample lon/lat along center rows/cols for tick labeling
    x_ticks = np.arange(x_dim)
    x_labels = np.round(to_np(lon2d[lat2d.shape[0]//2, :]), 2)

    y_ticks = np.arange(y_dim)
    y_labels = np.round(to_np(lat2d[:, lon2d.shape[1]//2]), 2)


    # Set axis ticks and labels
    ax.set_xticks(x_ticks[::10])
    ax.set_xticklabels(x_labels[::10])
    ax.set_xlabel("Longitude")


    ax.set_yticks(y_ticks[::10])
    ax.set_yticklabels(y_labels[::10])
    ax.set_ylabel("Latitude")
    
    
    x_min, x_max = lon_inds.min(), lon_inds.max()
    y_min, y_max = lat_inds.min(), lat_inds.max()
    z_min, z_max = z_inds.min(), z_inds.max()
    
    ax.set_xlim(x_min - 5, (x_max + 5))
    ax.set_ylim(y_min - 5, (y_max + 5))
    ax.set_zlim(0, (z_max + 3))
    
    ax.set_zlabel("Z (grid)")
    ax.set_title(f"3D Cloud Structure in Grid Space (Psuedo lat/lon) at {matched_time}")

    plt.tight_layout()
    plt.savefig(savepath + f"3Dcloudvox_ht{ht_level}T{threshold}_A{SIMULATION}D{domain}_{timestamp_str}.png")
    plt.show() if SHOW_FIGS else plt.close()


def plot_3d_voxel_mixingratio(name, var, mask_cases):

    # Colors to cycle through for each field (distinct from cloud color)
    field_colors = {
        "GRAUPEL": "orange",
        "SNOW": "purple",
        "ICE": "green",
        "HAIL": "red",
        "WATER VAPOR": "yellow"
    }


    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Print original shape (z, y, x)
    z_dim, y_dim, x_dim = mask_cases["cloud"].shape
    print("Original cloud mask shape (z, y, x):", z_dim, y_dim, x_dim)

    # Transpose to match (x, y, z) voxel format
    cloud_mask_vox = np.transpose(mask_cases["cloud"], (2, 1, 0)) # Downsample here if needed (e.g., [::2, ::2, ::2])

    print("Voxel shape (x, y, z):", cloud_mask_vox.shape)

    # *** New Mixing Ratio Check ***
    cloud_mask_vox = np.transpose(mask_cases["cloud"], (2, 1, 0))
    variable_mask = np.transpose((mask_cases["cloud"]) & (var > var.mean()), (2, 1, 0))

    # Plot voxels
    ax.voxels(variable_mask, facecolors=field_colors.get(name, "black"),alpha=0.6,label="Cloud")
    ax.voxels(cloud_mask_vox, facecolors='lightblue', edgecolor=None, alpha=0.4,label=f"{name} > {var.mean()} g/kg")
    
    # Axes and appearance
    ax.view_init(elev=10, azim=-90)

    
    x_min, x_max = lon_inds.min(), lon_inds.max()
    y_min, y_max = lat_inds.min(), lat_inds.max()
    z_min, z_max = z_inds.min(), z_inds.max()
    
    ax.set_xlim(x_min - 5, (x_max + 5))
    ax.set_ylim(y_min - 5, (y_max + 5))
    ax.set_zlim(0, (z_max + 3))
    
    ax.set_xlabel("X (grid)")
    ax.set_ylabel("Y (grid)")
    ax.set_zlabel("Z (grid)")
    ax.set_title(f"3D Cloud Structure (Grid Space) at {matched_time}")

    plt.tight_layout()
    plt.savefig(savepath + f"3Dcloudvox_ht{ht_level}T{threshold}_A{SIMULATION}D{domain}_{timestamp_str}.png")
    plt.show() if SHOW_FIGS else plt.close()

def plot_3d_scatter(ht_agl,mask_cases):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Convert WRF lat/lon to NumPy arrays
    lats, lons = latlon_coords(max_dbz)
    lats, lons = to_np(lats), to_np(lons)

    # Get indices of cloud voxels
    z_inds, y_inds, x_inds = np.where(mask_cases["cloud"])

    # Extract geographic + height values
    lat_vals = lats[y_inds, x_inds]
    lon_vals = lons[y_inds, x_inds]
    height_vals = ht_agl[z_inds, y_inds, x_inds]  # meters AGL

    # Sanity checks
    print("\nMin height:", np.min(height_vals))
    print("Max height:", np.max(height_vals))

    # Plot point cloud
    ax.scatter(lon_vals, lat_vals, height_vals, s=2, c='cyan', alpha=.8)

    # Tidy appearance
    ax.view_init(elev=10, azim=-90)
    ax.tick_params(axis='y', pad=10)
    ax.tick_params(axis='z', pad=10)
    ax.set_yticks([])
    ax.set_xlabel("Longitude")
    ax.set_zlabel("Height AGL (m)", labelpad=15)
    ax.set_title(f"3D Cloud Structure with Real Coordinates at {matched_time}")

    plt.tight_layout()
    plt.savefig(savepath + f"3Dcloudscat_ht{ht_level}T{threshold}_A{SIMULATION}D{domain}_{timestamp_str}.png")
    plt.show() if SHOW_FIGS else plt.close()



def identify_connected_cloud(cloud_frac, ht, ht_agl, max_dbz, fed, threshold, ht_level, x_y):
    """
    Identifies a connected cloud region starting from either the peak dBZ (if enabled) or a manually specified grid location.
    The region is grown vertically from the base level and checked for association with lightning activity.

    Parameters:
    -----------
    cloud_frac : xarray.DataArray
        3D cloud fraction array (z, y, x) — fraction of grid box covered by cloud [0 to 1].

    ht : xarray.DataArray
        3D array of height above sea level (m) for each grid point.

    ht_agl : xarray.DataArray
        3D array of height above ground level (m) for each grid point.

    max_dbz : xarray.DataArray
        3D reflectivity (dBZ) field used to find the cloud top intensity and derive the starting point if auto-detection is enabled.

    fed : xarray.DataArray or np.ndarray
        3D Flash Extent Density (FED) array used to determine whether lightning is present within the identified cloud region.

    threshold : float
        Cloud fraction threshold (e.g., 0.2 means 20% of grid cell must be cloud-filled).

    ht_level : int
        Vertical index (z-level) to use as the base for cloud region identification.

    x_y : tuple of int
        (x, y) grid indices to manually specify a starting point if auto-selection is disabled, converted from lat_lon.

    Returns:
    --------
    mean_start_height : float
        Mean height (m ASL) of the cloud region at the base level.

    mean_cloud_height : float
        Mean height (m ASL) of the entire 3D cloud region.

    z_inds, lat_inds, lon_inds : np.ndarray
        Arrays of the vertical, latitude (y), and longitude (x) indices comprising the connected cloud region.

    had_lightning : bool
        True if lightning (FED > 0) was detected in the identified cloud region; False otherwise.
    """
    base_level = ht_level
    snow_band_2d = (cloud_frac[base_level] >= threshold) & (dbz[base_level] > 0)
    labeled_2d, num_features = label(snow_band_2d)
    props = regionprops(labeled_2d)

    region_mask_2d = None
    start_label = 0

    if USE_MAX_DBZ:
        # Define region to search (Lake Ontario/LEE area)
        lat_min, lat_max = 43, 44
        lon_min, lon_max = -78.5, -75.5

        dbz_base = dbz.isel(bottom_top=base_level)
        lats, lons = latlon_coords(dbz_base)

        region_mask = (lats >= lat_min) & (lats <= lat_max) & \
                      (lons >= lon_min) & (lons <= lon_max)
        dbz_2d = dbz_base.where(region_mask)
        flat_dbz = dbz_2d.values.flatten()
        sorted_indices = np.argsort(flat_dbz)[::-1]

        ny, nx = dbz_2d.shape  # shape of the 2D grid after masking

        check_num = 0
        for idx in sorted_indices:
            # Skip masked values which are nan
            if np.isnan(flat_dbz[idx]):
                continue

            if flat_dbz[idx] < 25:
                print("!!! Starting location is not higher than 25 dBZ, skipping to the next time !!!")
                return "NO_CLOUD"
    
                '''
                # Convert flat index back to 2D indices
                j, i = np.unravel_index(idx, (ny, nx))
           
                t_850 = interplevel(tc, p, 850.0)  # Returns 2D array (y, x)
            
                # Now you can safely use j, i to access other variables
                sst_val = float(sst[j, i]) - 273.15
                t850_val = float(t_850[j, i])
                delta_t = sst_val - t850_val

                print(f"[j={j}, i={i}] SST = {sst_val:.2f} °C, T850 = {t850_val:.2f} °C, ΔT = {delta_t:.2f} °C, dBZ = {dbz_2d[j,i]}")

                
                '''

            if check_num >= 3:
                print("!!! Too many starting point checks, skipping to the next time !!!")
                return "NO_CLOUD"
            y_idx, x_idx = np.unravel_index(idx, dbz_2d.shape)
            start_label = labeled_2d[y_idx, x_idx]
            if start_label == 0:
                print("dBZ peak not inside any connected cloud region, checking new region using next highest reflectivty value.")
                check_num += 1
                continue

            region = props[start_label - 1]
            size = region.area
            print(f"Connected cloud size: {size} gridpoints")

            if not (min_gridboxes <= size <= max_gridboxes):
                print("Region size does not meet LLAP size criteria, checking new region using next highest reflectivity value")
                check_num += 1
                continue

            region_mask_2d = (labeled_2d == start_label)
            break
    else:
        # Use manually specified location
        start_label = labeled_2d[x_y[1], x_y[0]]
        if start_label != 0:
            region_mask_2d = (labeled_2d == start_label)

    if region_mask_2d is None:
        print("!!! Starting point not in any region, skipping to the next time !!!")
        return "NO_CLOUD"

    # Grow cloud region vertically 
    cloud_region_mask = np.zeros_like(cloud_frac, dtype=bool)
    cloud_region_mask[base_level] = region_mask_2d

    # Grow upward
    prev_layer = region_mask_2d.copy()
    for z in range(base_level + 1, cloud_frac.shape[0]):
        current_layer = (cloud_frac[z] >= threshold) & prev_layer
        if not np.any(current_layer):
            break
        cloud_region_mask[z] = current_layer
        prev_layer = current_layer

    # Grow downward
    prev_layer = region_mask_2d.copy()
    for z in range(base_level - 1, -1, -1):
        current_layer = (cloud_frac[z] >= threshold) & prev_layer
        if not np.any(current_layer):
            break
        cloud_region_mask[z] = current_layer
        prev_layer = current_layer

    #  Store masks 
    mask_cases["cloud"] = cloud_region_mask
    z_inds, lat_inds, lon_inds = np.where(cloud_region_mask)

    #  Flash mask (if applicable) 
    fed_np = fed if isinstance(fed, np.ndarray) else fed.values
    if np.any((fed_np > 0) & cloud_region_mask):
        had_lightning = True
        mask_cases["cloudlight"] = cloud_region_mask & (fed_np > 0)
        mask_cases["cloudnolight"] = cloud_region_mask & (fed_np == 0)
    else:
        had_lightning = False
        print("No flash pixels in cloud region — skipping flash mask creation.")

    #  Height statistics 
    if np.any(cloud_region_mask):
        mean_start_height = ht.isel(bottom_top=base_level).where(region_mask_2d).mean().item()
        mean_cloud_height = ht.where(cloud_region_mask).mean().item()
        mean_start_height_agl = ht_agl.isel(bottom_top=base_level).where(region_mask_2d).mean().item()
        mean_cloud_height_agl = ht_agl.where(cloud_region_mask).mean().item()

        print(f"\nBase level mean height: {mean_start_height:.1f} m | AGL: {mean_start_height_agl:.1f} m")
        print(f"Total cloud mean height: {mean_cloud_height:.1f} m | AGL: {mean_cloud_height_agl:.1f} m")
    else:
        print("No heights detected, check data")
        mean_start_height = float('nan')
        mean_cloud_height = float('nan')

    return (
        mean_start_height, mean_cloud_height,
        z_inds, lat_inds, lon_inds, had_lightning
    )

# #####################################
# Start looping through time (each file) #
# ######################################

for idx, filename in enumerate(filelist):
    failed_start = False

    matched_time = timelist[idx]
    matched_timeidx = timeidxlist[idx]

    timestamp_str = timelist[idx].strftime("%Y%m%d_%H%M%S") # For filenames

    print(f"Closest match: {matched_time} in file {filename} at time index {matched_timeidx}")

    # Get the WRF variables
    with Dataset(filename) as ds:

        # Convert desired coorindates to WRF gridbox coordinates
        x_y = ll_to_xy(ds, lat_lon[0], lat_lon[1])
        ht = getvar(ds, "z", timeidx=matched_timeidx)
        ht_agl = getvar(ds, "height_agl",timeidx=matched_timeidx)
        dbz = getvar(ds, "dbz", timeidx=matched_timeidx)
        max_dbz = getvar(ds, "mdbz", timeidx=matched_timeidx)
        p = getvar(ds, "pressure", timeidx=matched_timeidx)
        sst = getvar(ds, "SSTSK", timeidx=matched_timeidx)


        tc = getvar(ds, "tc", timeidx=matched_timeidx)
        lats = getvar(ds, "lat", timeidx=matched_timeidx)
        lons = getvar(ds, "lon",timeidx=matched_timeidx)
        cloud_frac = getvar(ds, "CLDFRA",timeidx=matched_timeidx) 
        land_mask = getvar(ds, "LANDMASK", timeidx=matched_timeidx)
        fed = getvar(ds,"LIGHTDENS", timeidx=matched_timeidx,meta=False)
        vert_velocity = getvar(ds, "wa",timeidx=matched_timeidx,meta=False)

        # Multiply by 1000 to go from kg/kg to g/kg while reading data in
        wv = getvar(ds, "QVAPOR", timeidx=matched_timeidx,meta=False) * 1000
        snow = getvar(ds, "QSNOW", timeidx=matched_timeidx,meta=False) * 1000
        ice = getvar(ds, "QICE", timeidx=matched_timeidx,meta=False) * 1000
        graupel  = getvar(ds, "QGRAUP", timeidx=matched_timeidx,meta=False) * 1000
        cloud = getvar(ds, "QCLOUD", timeidx=matched_timeidx, meta=False) * 1000
        hail = getvar(ds,"QHAIL", timeidx=matched_timeidx, meta=False) * 1000


    # Define dict where masks will be stored
    mask_cases = {}
    
    # Define dict of mixing ratios 
    mixing_ratios = {
        "SNOW": snow,
        "GRAUPEL": graupel,
        "ICE": ice,
        "WATER_VAPOR": wv,
        "CLOUD":cloud,
        "HAIL":hail
    }
   
    # Get cloud and flash masks, check if NO_CLOUD occured
    result = identify_connected_cloud(cloud_frac, ht, ht_agl, max_dbz, fed, threshold, ht_level, x_y)

    if result == "NO_CLOUD":
        print("!!! Skipping to next time !!!")
        continue  # Skip to the next time file in loop

    # Unpack our results
    (mean_start_height, mean_cloud_height, z_inds, lat_inds, lon_inds, had_lightning) = result
    
    # Ensure ht_agl is in NumPy only if absolutely necessary
    ht_agl = to_np(ht_agl)

    # Group definitions, use cloud but seperate based on lightning occurence
    groups = [
        ("cloud", cloud_all_with_lightning if had_lightning else cloud_all_without_lightning),
        ("cloudlight", cloud_with_flash),
        ("cloudnolight", cloud_without_flash),
        ("cloud", cloud_all)  # Final catch-all regardless of lightning
    ]

    # Loop through each group and store masked data
    for group_name, storage_dict in groups:
        mask = mask_cases.get(group_name)
        if mask is None:
            print(f"{group_name} not created")
            continue

        for name, var in mixing_ratios.items():
            storage_dict[name].append(var[mask])

        storage_dict["height_agl"].append(ht_agl[mask])
        # storage_dict["reflectivity"].append(max_dbz[mask])  # Uncomment if needed

    print("Values stored for clouds, cloud regions with lightning, and cloud regions without lightning")

    # === Stats and Histograms for Cloud ===
    for case_name, mask in mask_cases.items():
        print(f"\n===== STATS FOR {case_name.upper()} =====")
        print(f"\n{case_name.upper()} has {len(lat_inds)} grid points")

        for name, var in mixing_ratios.items():
            flat_data = var[mask]
            flat_data = flat_data[~np.isnan(flat_data)]
            flat_data_compressed = flat_data.compressed()
            
            # Optional: print stats
            print(f"\n{name} Mean: {np.ma.mean(flat_data_compressed):.3f}, Median: {np.ma.median(flat_data_compressed):.3f}, Std: {np.ma.std(flat_data_compressed):.3f}")
           
            # Prompt user for each type
            prompt_plot(f"Plot {name} histogram for {case_name}?", lambda: plot_q_histogram(var, mask))
            prompt_plot(f"Plot {name} 2D histogram by height for {case_name}?", lambda: plot_q_2d_histogram(var, mask))
    
    prompt_plot(f"Plot vertical velocity histogram for {case_name}?", lambda: plot_w_histogram(vert_velocity, mask))
    prompt_plot("Plot plan view of the cloud highlighting flash regions?", lambda: plot_plan_cloud(max_dbz, mask_cases))
    prompt_plot("Plot simulated composite reflectivity?", lambda: plot_mdbz(max_dbz, mask_cases))
    prompt_plot(f"Plot 3D scatter of the cloud?", lambda: plot_3d_scatter(ht_agl,mask_cases))

    if INTERACTIVE == True:
        prompt_plot(f"Plot 3D voxels of the cloud?", lambda: plot_3d_voxel(vert_velocity, land_mask, mask_cases))
        for name, data in mixing_ratios.items():
            prompt_plot(f"Plot 3D Voxels highlightning above average {name}?", lambda: plot_3d_voxel_mixingratio(name, data, mask_cases))
    else:
        print("Skipping Voxel plot to save time")

       
    
# === Summary statistics collector ===
summary_stats = []
raw_data_rows = []

all_groups = {
    "cloud_with_flash": cloud_with_flash,
    "cloud_without_flash": cloud_without_flash,
    "cloud_all_with_lightning": cloud_all_with_lightning,
    "cloud_all_without_lightning": cloud_all_without_lightning,
    "cloud_all": cloud_all,
}

# === Per-time-step stats ===
for group_name, group_dict in all_groups.items():
    n_timesteps = len(timelist)

    for i in range(n_timesteps):
        current_time = timelist[i]
        for var_name, var_list in group_dict.items():
            try:
                values = var_list[i]
            except IndexError:
                continue  # Skip this variable 
            stats = {
                "Group": group_name,
                "Time": current_time,
                "Variable": var_name,
                "Mean": np.mean(values),
                "Median": np.median(values),
                "Std Dev": np.std(values),
                "Min": np.min(values),
                "Max": np.max(values),
            }
            summary_stats.append(stats)

# === All-time stats ===
for group_name, group_dict in all_groups.items():
    for var_name, var_list in group_dict.items():

        # Concatenate all time slices
        try:
            all_values = np.concatenate(var_list)
        except ValueError:
            continue  # Skip if variable is empty across all times

        if all_values.size == 0:
            continue

        stats = {
            "Group": group_name,
            "Time": f"ALL",  # Label for all-time stats
            "Variable": var_name,
            "Mean": np.mean(all_values),
            "Median": np.median(all_values),
            "Std Dev": np.std(all_values),
            "Min": np.min(all_values),
            "Max": np.max(all_values),
        }
        summary_stats.append(stats)

# === Raw data storing ===
for group_name, group_dict in all_groups.items():
    # Get height_agl list once for this group
    heights_list = group_dict.get("height_agl", None)

    for var_name, var_list in group_dict.items():
        if var_name in ["time", "height_agl"]:
            continue

        for i, values in enumerate(var_list):
            current_time = timelist[i]
            
            # Get matching height_agl values
            if heights_list is None or i >= len(heights_list):
                continue
            height_values = heights_list[i]

            # Sanity check: matching lengths
            if len(values) != len(height_values):
                continue  # Skip if misaligned

            # Store both value and height per gridpoint
            for val, h in zip(values, height_values):
                raw_data_rows.append({
                    "Group": group_name,
                    "Time": current_time,
                    "Variable": var_name,
                    "Value": val,
                    "Height_AGL": h
                })




# === Create DataFrame ===
df_stats = pd.DataFrame(summary_stats)
df_raw = pd.DataFrame(raw_data_rows)

# === Optional: Sort for clean output ===
df_stats.sort_values(by=["Group", "Variable", "Time"], inplace=True)

# === Print and Save ===
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:.3f}".format)

print("\n=== Summary Statistics for Cloud Variables ===\n")
print(df_stats.to_string(index=False))

# === Adjust time for filename
filetime_start, filetime_end = start_time.strftime("%Y%m%d%H%M"), end_time.strftime("%Y%m%d%H%M"),

df_stats.to_csv(os.path.join(savepath, f"cloud_stats_{filetime_start}_{filetime_end}.csv"), index=False)
print(f"Saved histograms and summary statistics in: {savepath}")

df_raw.to_csv(os.path.join(savepath, f"cloud_var_rawdata_{filetime_start}_{filetime_end}.csv"), index=False)






