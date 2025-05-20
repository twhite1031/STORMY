from multiprocessing import Pool
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  
import numpy as np
import pyart
import radarfuncs
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
import matplotlib.patches as mpatches

# Load dataset
df = pd.read_csv('/data2/white/DATA/MET399/morphs/hydro_with_morphology.csv',index_col=0, parse_dates=True)

# List of hydrometeor columns
hydro_cols = ['ND','BI','GC','IC','DS','WS','RA','HR','BD','GR','HA','LH','GH','--','UK','RH']
morph_types = df['Morphology'].unique()

# Build list of input args
input_args = [(idx, row) for idx, row in df.iterrows()]

# Initialize cumulative_counts
cumulative_counts = {morph: None for morph in morph_types}

def process_single_radar(args):
    idx, row = args  # unpack tuple
    if idx.minute != 0:
        return None  # skip non-hourly

    morph = row['Morphology']
    event = row['Event']
    radar_data_dir = f"/data2/white/DATA/MET399/NEXRADLVL2/{event}/"

    radar_file = radarfuncs.find_closest_radar_file(idx, radar_data_dir, "KTYX")
    if not radar_file:
        return None

    try:
        radar = pyart.io.read(radar_file)
        grid = pyart.map.grid_from_radars(
            radar,
            grid_shape=(1, 500, 500),
            grid_limits=((1000, 1000), (-250000, 250000), (-250000, 250000))
        )
        refl_grid = grid.fields['reflectivity']['data'][0]
        
      
        presence_mask = (~np.ma.getmaskarray(refl_grid)) & (refl_grid.data > 5) & (refl_grid.data < 50)

        
        return morph, presence_mask.astype(int)

    except Exception as e:
        print(f"Error processing {radar_file}: {e}")
        return None

with Pool(processes=15) as pool:  # adjust number of processes
    results = pool.map(process_single_radar, input_args)

for result in results:
    if result is None:
        continue
    morph, presence_mask = result
    if cumulative_counts[morph] is None:
        cumulative_counts[morph] = presence_mask
        cumulative_counts[morph]
    else:
        cumulative_counts[morph] += presence_mask

# Extract grided data from sample file 
radar_data_dir = "/data2/white/DATA/MET399/NEXRADLVL2/20181110/"
sample_idx = datetime(2018,11,10,14,0)
radar_file = radarfuncs.find_closest_radar_file(sample_idx, radar_data_dir, "KTYX")
print("Radar file: ", radar_file)
radar = pyart.io.read(radar_file)

grid = pyart.map.grid_from_radars(
    radar,
    grid_shape=(1, 500, 500),
    grid_limits=((1000, 1000), (-250000, 250000), (-250000, 250000))
)
# Example of lat/lon
lat_grid = grid.point_latitude['data'][0]  # 0th vertical level
lon_grid = grid.point_longitude['data'][0]

# 4️⃣ Stack counts into array
morph_list = list(cumulative_counts.keys())
count_stack = np.stack([cumulative_counts[morph] for morph in morph_list], axis=-1)  # shape (ny, nx, nmorph)

dominant_index = np.argmax(count_stack, axis=-1)

print("Dominant index unique values:", np.unique(dominant_index))

print("Count stack shape:", count_stack.shape)
print("Count stack max:", np.max(count_stack))
print("Count stack sum:", np.sum(count_stack))
color_map = {
    'LLAP': [1, 0, 0],        # red
    'Broad Coverage': [0, 1, 0],       # green
    'Orographic': [0, 0, 1],  # blue
    'Hybrid': [1, 0, 1],      # magenta
}

color_array = np.array([color_map[morph] for morph in morph_list])  # shape (nmorph, 3)
rgb_image = color_array[dominant_index]  # shape (ny, nx, 3)

total_counts = np.sum(count_stack, axis=-1)

mask = (total_counts == 0)

rgb_image[mask] = [1.0, 1.0, 1.0]  # assign white for no detections

print("color_array:", color_array)
print("RGB image min:", np.min(rgb_image))
print("RGB image max:", np.max(rgb_image))
print("RGB image shape:", rgb_image.shape)

# Attempted to try and blend colors instead of choosing dominant morphology here

#color_array = np.array([color_map[morph] for morph in morph_list])  # shape (nmorph, 3)
# normalize counts to fractions per pixel
#total_counts = np.sum(count_stack, axis=-1, keepdims=True)
#fraction_stack = count_stack / np.maximum(total_counts, 1)  # avoid divide by zero

# weighted sum of colors
#rgb_image = np.einsum('ijk,kl->ijl', fraction_stack, color_array)  # shape (ny, nx, 3)

# optional: compute opacity as total detection frequency normalized
#opacity = total_counts.squeeze(-1) / total_counts.max()
#alpha_channel = opacity[..., np.newaxis]
#rgba_image = np.concatenate((rgb_image, alpha_channel), axis=-1)  # shape (ny, nx, 4)

#print("RGBA min/max:", np.min(rgba_image), np.max(rgba_image))
#print("Alpha min/max:", np.min(rgba_image[..., 3]), np.max(rgba_image[..., 3]))

#print("Lon bounds:", lon_grid.min(), lon_grid.max())
#print("Lat bounds:", lat_grid.min(), lat_grid.max())

#plt.figure(figsize=(12,8))
#ax = plt.axes(projection=ccrs.PlateCarree())

# optional: zoom map to radar domain
#ax.set_extent([lon_grid.min(), lon_grid.max(), lat_grid.min(), lat_grid.max()], crs=ccrs.PlateCarree())


#mesh = ax.imshow(rgba_image, extent=(lon_grid.min(), lon_grid.max(), lat_grid.min(), lat_grid.max()), origin='lower',transform=ccrs.PlateCarree())
# add basemap features
#ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='none', edgecolor='black')
#ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
#ax.add_feature(cfeature.BORDERS.with_scale('10m'))
#ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':')


#plt.colorbar(mesh, ax=ax, label='Opacity = frequency (0-1)')
#plt.title('Morphology Frequency Blended Heatmap over Lake Ontario')
#plt.show()

# ---------------------------------------------------

plt.figure(figsize=(12,8))
ax = plt.axes(projection=ccrs.PlateCarree())

# optional: zoom map to radar domain
ax.set_extent([lon_grid.min(), lon_grid.max(), lat_grid.min(), lat_grid.max()], crs=ccrs.PlateCarree())

LLAP_counts = cumulative_counts["LLAP"]

# Attempting to set 0 as white
cmap = plt.cm.Reds.copy()
cmap.set_under('white')

mesh = ax.pcolormesh(
    lon_grid,
    lat_grid,
    LLAP_counts,
    cmap=cmap,
    shading='auto',
    vmin=1,
    vmax=LLAP_counts.max(),
    transform=ccrs.PlateCarree()
)

ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='none', edgecolor='black')
ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
ax.add_feature(cfeature.BORDERS.with_scale('10m'))
ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':')

plt.colorbar(mesh, ax=ax, label='Hours')
plt.title('LLAP Frequency Heatmap over Lake Ontario')

plt.figure(figsize=(12,8))
ax = plt.axes(projection=ccrs.PlateCarree())

# optional: zoom map to radar domain
ax.set_extent([lon_grid.min(), lon_grid.max(), lat_grid.min(), lat_grid.max()], crs=ccrs.PlateCarree())

BC_counts = cumulative_counts["Broad Coverage"]

# Attempting to set 0 as white
cmap = plt.cm.Reds.copy()
cmap.set_under('white')

mesh = ax.pcolormesh(
    lon_grid,
    lat_grid,
    BC_counts,
    cmap=cmap,
    shading='auto',
    vmin=1,
    vmax=BC_counts.max(),
    transform=ccrs.PlateCarree()
)

ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='none', edgecolor='black')
ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
ax.add_feature(cfeature.BORDERS.with_scale('10m'))
ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':')

plt.colorbar(mesh, ax=ax, label='Hours')
plt.title('Broad Coverage Frequency Heatmap over Lake Ontario')


plt.figure(figsize=(12,8))
ax = plt.axes(projection=ccrs.PlateCarree())

# optional: zoom map to radar domain
ax.set_extent([lon_grid.min(), lon_grid.max(), lat_grid.min(), lat_grid.max()], crs=ccrs.PlateCarree())

Oro_counts = cumulative_counts["Orographic"]

# Attempting to set 0 as white
cmap = plt.cm.Reds.copy()
cmap.set_under('white')

mesh = ax.pcolormesh(
    lon_grid,
    lat_grid,
    Oro_counts,
    cmap=cmap,
    shading='auto',
    vmin=1,
    vmax=Oro_counts.max(),
    transform=ccrs.PlateCarree()
)

ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='none', edgecolor='black')
ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
ax.add_feature(cfeature.BORDERS.with_scale('10m'))
ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':')

plt.colorbar(mesh, ax=ax, label='Hours')
plt.title('Orographic Frequency Heatmap over Lake Ontario')


plt.figure(figsize=(12,8))
ax = plt.axes(projection=ccrs.PlateCarree())

# optional: zoom map to radar domain
ax.set_extent([lon_grid.min(), lon_grid.max(), lat_grid.min(), lat_grid.max()], crs=ccrs.PlateCarree())

HYB_counts = cumulative_counts["Hybrid"]

# Attempting to set 0 as white
cmap = plt.cm.Reds.copy()
cmap.set_under('white')

mesh = ax.pcolormesh(
    lon_grid,
    lat_grid,
    HYB_counts,
    cmap=cmap,
    shading='auto',
    vmin=1,
    vmax=HYB_counts.max(),
    transform=ccrs.PlateCarree()
)

ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='none', edgecolor='black')
ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
ax.add_feature(cfeature.BORDERS.with_scale('10m'))
ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':')

plt.colorbar(mesh, ax=ax, label='Hours')
plt.title('Hybrid Frequency Heatmap over Lake Ontario')

plt.figure(figsize=(12,8))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([lon_grid.min(), lon_grid.max(), lat_grid.min(), lat_grid.max()], crs=ccrs.PlateCarree())
mesh = ax.pcolormesh(
    lon_grid,
    lat_grid,
    np.zeros_like(lon_grid),  # dummy data
    color=rgb_image.reshape(-1, 3),
    shading='auto',
    transform=ccrs.PlateCarree()
)
#ax.imshow(
#    rgb_image,
#    extent=(lon_grid.min(), lon_grid.max(), lat_grid.min(), lat_grid.max()),
#    origin='lower',
#    transform=ccrs.PlateCarree()
#)
ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='none', edgecolor='black')
ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
ax.add_feature(cfeature.BORDERS.with_scale('10m'))
ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':')

legend_handles = [
    mpatches.Patch(color=color_map['LLAP'], label='LLAP'),
    mpatches.Patch(color=color_map['Broad Coverage'], label='Broad Coverage'),
    mpatches.Patch(color=color_map['Orographic'], label='Orographic'),
    mpatches.Patch(color=color_map['Hybrid'], label='Hybrid')
]

# Add legend to axis
ax.legend(handles=legend_handles, loc='upper right', title='Morphology')
ax.set_title('Dominant Morphology Map')

plt.show()

