import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Define bounding box coordinates
lat_min, lat_max = 43.25, 44.25
lon_min, lon_max = -76.25, -75.25

# Define box corners
lats = [lat_min, lat_max, lat_max, lat_min, lat_min]
lons = [lon_min, lon_min, lon_max, lon_max, lon_min]

# Create plot
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([lon_min - 1, lon_max + 1, lat_min - 1, lat_max + 1])

# Add features
ax.add_feature(cfeature.STATES, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--')

# Plot bounding box
ax.plot(lons, lats, 'b-', linewidth=2, transform=ccrs.PlateCarree(), label='Bounding Box')
ax.scatter([lon_min, lon_max, lon_max, lon_min], 
           [lat_min, lat_min, lat_max, lat_max], 
           color='red', label='Corners', transform=ccrs.PlateCarree())

# Labels and legend
ax.set_title('Bounding Box on USA Map')
ax.legend()


# Show plot
plt.show()

