import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  
import numpy as np
import pyart
import radarfuncs
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Load dataset
df = pd.read_csv('/data2/white/DATA/MET399/morphs/hydro_with_morphology.csv',index_col=0, parse_dates=True)

# List of hydrometeor columns
hydro_cols = ['ND','BI','GC','IC','DS','WS','RA','HR','BD','GR','HA','LH','GH','--','UK','RH']
morph_types = df['Morphology'].unique()

# Group by Morphology and calculate mean frequencies
morph_avg = df.groupby('Morphology')[hydro_cols].mean()

# You'll determine shape when loading the first radar file
cumulative_counts = {}
for morph in morph_types:
    cumulative_counts[morph] = None  # placeholder

for morph in morph_types:
    cumulative_counts[morph] = None  # placeholder
for idx, row in df.iterrows():
    if idx.minute == 00:
        event_time = str(row['Event'])  # e.g., '20181110'
        morph = row['Morphology']
        event = row['Event']
        radar_data_dir = f"/data2/white/DATA/MET399/NEXRADLVL2/{event}/"

        dt = idx # Index is a datetime
        # Build filename based on event_time
        radar_file = radarfuncs.find_closest_radar_file(dt, radar_data_dir, "KTYX")

        # Load radar file
        if not radar_file:
            continue
        else:
            radar = pyart.io.read(radar_file)


        grid = pyart.map.grid_from_radars(radar,grid_shape=(1, 500, 500),grid_limits=((1000, 1000), (-250000, 250000), (-250000, 250000)))
        refl_grid = grid.fields['reflectivity']['data'][0]  # 0th vertical level
      
        # Project reflectivity to Cartesian grid (optional, if in polar coordinates)
        # You may need pyart.map.grid_from_radars() here → see note below

        # Initialize storage array if first file
        if cumulative_counts[morph] is None:
            cumulative_counts[morph] = np.zeros_like(refl_grid)
        
        # 1 for valid reflectivity, 0 for masked/no data
        presence_mask = ~np.ma.getmaskarray(refl_grid)

        # accumulate presence (frequency count) instead of reflectivity sum
        cumulative_counts[morph] += presence_mask.astype(int)


# Example of lat/lon
lat_grid = grid.point_latitude['data'][0]  # 0th vertical level
lon_grid = grid.point_longitude['data'][0]

plt.figure(figsize=(10,6))
plt.pcolormesh(lon_grid, lat_grid, cumulative_counts['LLAP'], cmap='viridis')
plt.colorbar(label='Frequency of reflectivity detection')
plt.xlabel('Distance East (km)')
plt.ylabel('Distance North (km)')
plt.title('Reflectivity detection frequency – LLAP morphology')
plt.show()


# 4️⃣ Stack counts into array
morph_list = list(cumulative_counts.keys())
count_stack = np.stack([cumulative_counts[morph] for morph in morph_list], axis=-1)  # shape (ny, nx, nmorph)

color_map = {
    'LLAP': [1, 0, 0],        # red
    'Broad': [0, 1, 0],       # green
    'Orographic': [0, 0, 1],  # blue
    'Hybrid': [1, 0, 1],      # magenta
}

color_array = np.array([color_map[morph] for morph in morph_list])  # shape (nmorph, 3)

# normalize counts to fractions per pixel
total_counts = np.sum(count_stack, axis=-1, keepdims=True)
fraction_stack = count_stack / np.maximum(total_counts, 1)  # avoid divide by zero

# weighted sum of colors
rgb_image = np.einsum('ijk,kl->ijl', fraction_stack, color_array)  # shape (ny, nx, 3)

# optional: compute opacity as total detection frequency normalized
opacity = total_counts.squeeze(-1) / total_counts.max()
alpha_channel = opacity[..., np.newaxis]

rgba_image = np.concatenate((rgb_image, alpha_channel), axis=-1)  # shape (ny, nx, 4)

plt.figure(figsize=(12,8))

mesh = ax.pcolormesh(
    lon_grid, lat_grid, normalized_frequency,  # normalized_frequency = 0–1 array or rgba_image[..., 3]
    color=rgba_image.reshape(-1,4),  # flatten colors
    shading='auto',
    transform=ccrs.PlateCarree()
)

# add basemap features
ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='none', edgecolor='black')
ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
ax.add_feature(cfeature.BORDERS.with_scale('10m'))
ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':')

# optional: zoom map to radar domain
ax.set_extent([lon_grid.min(), lon_grid.max(), lat_grid.min(), lat_grid.max()], crs=ccrs.PlateCarree())

plt.colorbar(mesh, ax=ax, label='Opacity = frequency (0-1)')
plt.title('Morphology Frequency Blended Heatmap over Lake Ontario')
plt.show()
