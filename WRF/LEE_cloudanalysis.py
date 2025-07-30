import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import from_levels_and_colors
import matplotlib.patches as patches
from cartopy import crs
from netCDF4 import Dataset
from wrf import (getvar, to_np, get_cartopy, latlon_coords, vertcross,
                 interpline, CoordPair, xy_to_ll,ll_to_xy,cartopy_xlim, cartopy_ylim)
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
"""
A script used to automatically identify Lake-effect band locations based on connected cloud fractions at a certain level, 
then classifying the region as a cloud band for analysis
"""

# --- USER INPUT ---
wrf_date_time = datetime(2022,11,18,13,50,00)
domain = 2

INTERACTIVE = True  # Set to False for non-interactive (auto-run) mode
SHOW_FIGS = True # Do not display completed figures
USE_MAX_DBZ = True # Toggle this to switch between lat/lon and max dBZ

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

timestamp_str = matched_time.strftime("%Y%m%d_%H%M%S")

# Get the WRF variables
with Dataset(matched_file) as ds:

    # Convert desired coorindates to WRF gridbox coordinates
    x_y = ll_to_xy(ds, lat_lon[0], lat_lon[1])
    ht = getvar(ds, "z", timeidx=matched_timeidx)
    ht_agl = getvar(ds, "height_agl",timeidx=matched_timeidx)
    dbz = getvar(ds, "dbz", timeidx=matched_timeidx)
    max_dbz = getvar(ds, "mdbz", timeidx=matched_timeidx)
    Z = 10**(dbz/10.) # Use linear Z for interpolation
    lats = getvar(ds, "lat", timeidx=matched_timeidx)
    lons = getvar(ds, "lon",timeidx=matched_timeidx)
    cloud_frac = getvar(ds, "CLDFRA",timeidx=matched_timeidx) # Assuming CLDFRA is at level 10
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

# Function to easily "scroll" through figures
def show_fig_with_keypress(key='enter'):
    fig = plt.gcf()
    ax = plt.gca()
    pressed = plt.waitforbuttonpress()
    plt.close()

def identify_connected_cloud(cloud_frac, ht, ht_agl, max_dbz, fed, threshold, ht_level, x_y):
    base_level = ht_level
    snow_band_2d = cloud_frac[base_level, :, :] >= threshold
    labeled_2d, num_features = label(snow_band_2d)
    labels, counts = np.unique(labeled_2d, return_counts=True)
    label_counts = dict(zip(labels, counts))
    props = regionprops(labeled_2d)

    if USE_MAX_DBZ:
        # Isolate Lake Ontario (LEE) Region
        lat_min, lat_max = 43, 44
        lon_min, lon_max = -78.5, -75.5
        dbz_base = dbz.isel(bottom_top=base_level)
        lats, lons = latlon_coords(dbz_base)

        region_mask = (lats >= lat_min) & (lats <= lat_max) & \
                      (lons >= lon_min) & (lons <= lon_max)
                    

        dbz_2d = dbz_base.where(region_mask)
        flat_dbz = dbz_2d.values.flatten()
        sorted_indices = np.argsort(flat_dbz)[::-1]
        
        check_num = 0
        for idx in sorted_indices:

            # Make sure our starting points are higher than 30 dBZ
            if flat_dbz[idx] < 25:
                print("!!! Starting location is not higher than 25 dBZ, skipping to next time !!!")
                return "NO_CLOUD"
            
            if check_num == 3:
                print("!!! Too many starting point checks, skipping to next time !!!")
                return "NO_CLOUD"
                
            if np.isnan(flat_dbz[idx]):
                continue

            y_idx, x_idx = np.unravel_index(idx, dbz_2d.shape)
            start_label = labeled_2d[y_idx, x_idx]

            if start_label == 0:
                print("dBZ peak not inside any connected cloud region.")
                check_num += 1
                continue  # Not part of a cloud region at this level

            size = label_counts[start_label]
            print(f"Connected cloud size: {size} gridpoints")
            
            if not (min_gridboxes <= size <= max_gridboxes):
                print("Region size does not meet LLAP size criteria.")
                check_num += 1
                continue
            
            region = props[start_label - 1]  # regionprops is 0-indexed, but label starts at 1

            # Compute aspect ratio (major / minor axis lengths)
            if region.minor_axis_length == 0:
                aspect_ratio = np.inf  # or skip
            else:
                aspect_ratio = region.major_axis_length / region.minor_axis_length

            print(f"Aspect ratio: {aspect_ratio:.2f}")

            '''
            # Filter for LLAP-style elongation, USUALLY OFF
            if aspect_ratio < aspect_ratio_thresh:
                print("Region not elongated enough for LLAP criteria.")
                check_num += 1
                continue
            '''

            # Optional: get coordinates
            with Dataset(matched_file) as ds:
                dbz_x_y = xy_to_ll(ds, x_idx, y_idx)

            print(f"LLAP or LES candidate at x={x_idx}, y={y_idx}, dBZ={flat_dbz[idx]}, size={size}, lat/lon={dbz_x_y.values}")
            break  # or keep collecting if you want all matches
    
    else:
        # Manual input to start search
        start_label = labeled_2d[x_y[1], x_y[0]]
        dbz_x_y = lat_lon

    cloud_region_mask = np.zeros_like(cloud_frac, dtype=bool)

    if start_label == 0:
        print("Starting point not in any region.")
        #sys.exit()
    else:
        region_mask_2d = (labeled_2d == start_label)
        cloud_region_mask[base_level, :, :] = region_mask_2d

        # Grow upward
        prev_layer = region_mask_2d.copy()
        for z in range(base_level + 1, cloud_frac.shape[0]):
            current_layer = (cloud_frac[z, :, :] >= threshold) & prev_layer
            if not np.any(current_layer):
                break
            cloud_region_mask[z, :, :] = current_layer
            prev_layer = current_layer

        # Grow downward
        prev_layer = region_mask_2d.copy()
        for z in range(base_level - 1, -1, -1):
            current_layer = (cloud_frac[z, :, :] >= threshold) & prev_layer
            if not np.any(current_layer):
                break
            cloud_region_mask[z, :, :] = current_layer
            prev_layer = current_layer

    # Store compled cloud mask
    mask_cases["cloud"] = cloud_region_mask

    # Index of cloud region
    z_inds, lat_inds, lon_inds = np.where(cloud_region_mask)

    # Convert FED if needed
    fed_np = fed if isinstance(fed, np.ndarray) else fed.values

    if np.any((fed_np > 0) & cloud_region_mask):
        flash_mask = cloud_region_mask & (fed_np > 0)
        no_flash_mask = cloud_region_mask & (fed_np == 0)

    # Add to dictionary or process them
        mask_cases["cloudlight"] = flash_mask
        mask_cases["cloudnolight"] = no_flash_mask
    else:
        print("No flash pixels in cloud region — skipping flash mask creation.")
       
    # Mean height calculations
    if np.any(cloud_region_mask):
        mean_start_height = ht.isel(bottom_top=base_level).where(region_mask_2d).mean().item()
        mean_cloud_height = ht.where(cloud_region_mask).mean().item()
        print(f"Base level mean height (level {base_level}): {mean_start_height:.1f} m")
        print(f"Total cloud region mean height: {mean_cloud_height:.1f} m")

        mean_start_height_agl = ht_agl.isel(bottom_top=base_level).where(region_mask_2d).mean().item()
        mean_cloud_height_agl = ht_agl.where(cloud_region_mask).mean().item()
        print(f"Base level mean height agl (level {base_level}): {mean_start_height_agl:.1f} m")
        print(f"Total cloud region mean height agl: {mean_cloud_height_agl:.1f} m")

    else:
        mean_start_height = float('nan')
        mean_cloud_height = float('nan')

    return (
        mean_start_height, mean_cloud_height,
        z_inds, lat_inds, lon_inds,dbz_x_y
    )

(mean_start_height, mean_cloud_height, z_inds, lat_inds, lon_inds,dbz_x_y) = identify_connected_cloud(cloud_frac, ht, ht_agl, max_dbz, fed, threshold, ht_level, x_y)
ht_agl = to_np(ht_agl)


# Define all mixing ratio variables
mixing_ratios = {
    "SNOW": snow,
    "GRAUPEL": graupel,
    "ICE": ice,
    "WATER VAPOR": wv,
    "CLOUD":cloud,
    "HAIL":hail
}

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
'''
# === Stats for Cloud ===
for case_name, mask in mask_cases.items():
    print(f"\n===== STATS FOR {case_name.upper()} =====")
    print(f"\n{case_name.upper()} has {len(lat_inds)} grid points")

    for name, var in mixing_ratios.items():
        flat_data = var[mask]
        flat_data = flat_data[~np.isnan(flat_data)]
        flat_data_compressed = flat_data.compressed()
        
        # Optional: print stats
        print(f"\n{name} Mean: {np.mean(flat_data_compressed):.3f}, Median: {np.median(flat_data_compressed):.3f}, Std: {np.std(flat_data_compressed):.3f}")

# === Histograms for Cloud ===

for case_name, mask in mask_cases.items():
    print(f"\n===== HISTOGRAM FOR {case_name.upper()} =====")
        
    for name, var in mixing_ratios.items():
        flat_data = var[mask]
        flat_data = flat_data[~np.isnan(flat_data)]

        def plot_q_histogram():
            plt.figure(figsize=(8, 10))
            plt.hist(flat_data, bins=50, edgecolor='black')
            plt.title(f"{name} Mixing Ratio Histogram (g/kg)")
            plt.xlabel("Mixing Ratio (g/kg)")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.suptitle(
                f"Mean height {mean_start_height:.1f}m | Level {ht_level} | Gridboxes: {len(lat_inds)} | T: {threshold} | "
                f"A{SIMULATION} D{domain}"
            )
            filename = f"{case_name}_hist{name[0:3]}_ht{ht_level}T{threshold}_A{SIMULATION}D{domain}_{timestamp_str}.png"
            plt.savefig(savepath + filename)
            plt.show() if SHOW_FIGS else plt.close()

        def plot_q_2d_histogram():
            z_levels = var.shape[0]

            # Clip upper limit at 99.5th percentile
            upper_limit = np.percentile(flat_data, 90) # Testing binning with perctiles
            mix_bins = np.linspace(0, upper_limit, 100)
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
            plt.title(f"{name} 2D Histogram ({case_name})")
            plt.colorbar(label="Frequency")
            plt.tight_layout()
            filename2d = f"{case_name}_2Dhist{name[0:3]}_ht{ht_level}T{threshold}_A{SIMULATION}D{domain}_{timestamp_str}.png"
            plt.savefig(savepath + filename2d)
            plt.show() if SHOW_FIGS else plt.close()

        # Prompt user for each type
        prompt_plot(f"Plot {name} histogram for {case_name}?", plot_q_histogram)
        prompt_plot(f"Plot {name} 2D histogram by height for {case_name}?", plot_q_2d_histogram)
        
    flat_data = vert_velocity[mask]
    flat_data = flat_data[~np.isnan(flat_data)]

    def plot_w_histogram():
        plt.figure(figsize=(8, 10))
        plt.hist(flat_data, bins=50, edgecolor='black')
        plt.title(f"Vertical Velocity Histogram (m/s)")
        plt.xlabel("Vertical Velocity (m/s)")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.suptitle(
            f"Mean height {mean_start_height:.1f}m | Level {ht_level} | Gridboxes: {len(lat_inds)} | T: {threshold} | "
            f"A{SIMULATION} D{domain}"
        )

        filename = f"{case_name}_histw_ht{ht_level}T{threshold}_A{SIMULATION}D{domain}_{timestamp_str}.png"
        plt.savefig(savepath + filename)
        plt.show() if SHOW_FIGS else plt.close()

    prompt_plot(f"Plot Vertical Velocity histogram for {case_name}?", plot_w_histogram)

     
# === 3D Scatter Plot (with real lat/lon/height) ===

def plot_3d_scatter():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Convert WRF lat/lon to NumPy arrays
    lats, lons = latlon_coords(dbz)
    lats, lons = to_np(lats), to_np(lons)

    # Get indices of cloud voxels
    z_inds, y_inds, x_inds = np.where(mask_cases["cloud"])

    # Extract geographic + height values
    lat_vals = lats[y_inds, x_inds]
    lon_vals = lons[y_inds, x_inds]
    height_vals = ht_agl[z_inds, y_inds, x_inds]  # meters AGL

    # Sanity checks
    print("Min height:", np.min(height_vals))
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
    ax.set_title("3D Cloud Structure with Real Coordinates")

    plt.tight_layout()
    plt.savefig(savepath + f"3Dcloudscat_ht{ht_level}T{threshold}_A{SIMULATION}D{domain}_{timestamp_str}.png")
    plt.show() if SHOW_FIGS else plt.close()

prompt_plot(f"Plot 3D scatter of the cloud?", plot_3d_scatter)

# === 3D Voxel Plot (grid-based, no lat/lon) ===

def plot_3d_voxel():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Print original shape (z, y, x)
    z_dim, y_dim, x_dim = mask_cases["cloud"].shape
    print("Original cloud mask shape (z, y, x):", z_dim, y_dim, x_dim)

    # Transpose to match (x, y, z) voxel format
    cloud_mask_vox = np.transpose(mask_cases["cloud"], (2, 1, 0)) # Downsample here if needed (e.g., [::2, ::2, ::2])

    print("Voxel shape (x, y, z):", cloud_mask_vox.shape)

    # *** New Updraft Check ***
    cloud_mask_vox = np.transpose(mask_cases["cloud"], (2, 1, 0))
    strong_updraft_mask = np.transpose((mask_cases["cloud"]) & (vert_velocity > 2.0), (2, 1, 0))

    # Plot voxels
    ax.voxels(strong_updraft_mask, facecolors='magenta',alpha=0.6)
    ax.voxels(cloud_mask_vox, facecolors='lightblue', edgecolor=None, alpha=0.4)
    
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
    ax.set_title("3D Cloud Structure (Grid Space)")

    plt.tight_layout()
    plt.savefig(savepath + f"3Dcloudvox_ht{ht_level}T{threshold}_A{SIMULATION}D{domain}_{timestamp_str}.png")
    plt.show() if SHOW_FIGS else plt.close()

prompt_plot(f"Plot 3D voxels of the cloud?", plot_3d_voxel)
'''
# === 3D Voxel Plot for Mixing Ratios
# Colors to cycle through for each field (distinct from cloud color)
field_colors = {
    "GRAUPEL": "orange",
    "SNOW": "purple",
    "ICE": "green",
    "HAIL": "red",
    "WATER VAPOR": "yellow"
}
    

def plot_3d_voxel_mixingratio(name, var):
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
    cloud_mask = mask_cases["cloud"]
    
    # Now take the mean only inside the cloud
    mean_val = var[cloud_mask].mean() if np.any(cloud_mask) else 0.0 
    # Create 3D mask of where mixing ratio is above mean, inside the cloud
    variable_mask = np.transpose((cloud_mask) & (var > mean_val), (2, 1, 0))
    
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
    ax.set_title(f"3D Cloud with above average {name} mixing ratios highlighted at {matched_time}")

    plt.savefig(savepath + f"3Dcloudvox{name}_ht{ht_level}T{threshold}_A{SIMULATION}D{domain}_{timestamp_str}.png")
    plt.show() if SHOW_FIGS else plt.close()

for name, data in mixing_ratios.items():
    prompt_plot(f"Plot 3D Voxels highlightning above average {name}?", lambda: plot_3d_voxel_mixingratio(name, data))


# === Plan View Map of Cloud and flash regions

def plot_plan_cloud():
    
    # Collapse cloud mask (this one should always exist)
    cloud_mask_2d = np.max(mask_cases["cloud"], axis=0)

    # Optional masks: flash_mask and no_flash_mask
    flash_mask_2d = np.max(mask_cases["cloudlight"], axis=0) if "cloudlight" in mask_cases else None
        
    # Get the lat/lon points 
    lats, lons = latlon_coords(dbz)

    # Get the cartopy projection object
    cart_proj = get_cartopy(dbz)

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

    # Plot Cloud Region (red)
    cf = ax.contourf(
        to_np(lons), to_np(lats), to_np(cloud_mask_2d.astype(float)),
        levels=[0.5, 1.5], colors=['red'], alpha=0.3,
        transform=crs.PlateCarree(), zorder=4
    )

    ax.scatter(dbz_x_y[1], dbz_x_y[0], color='red', marker='x', s=50, transform=crs.PlateCarree())

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
    
    # Add the gridlines
    gl = ax.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=False)
    gl.xlabel_style = {'rotation': 'horizontal','size': 14,'ha':'center'} # Change 14 to your desired font size
    gl.ylabel_style = {'size': 14}  # Change 14 to your desired font size
    gl.xlines = True
    gl.ylines = True

    gl.top_labels = False  # Disable top labels
    gl.right_labels = False  # Disable right labels
    gl.xpadding = 20

    # Final touches
    ax.set_xlim(cartopy_xlim(dbz))
    ax.set_ylim(cartopy_ylim(dbz))
    plt.title("Cloud region linked to flash")
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
    plt.show() if SHOW_FIGS else plt.close()

prompt_plot("Plot plan view of the cloud highlighting flash regions?", plot_plan_cloud)





