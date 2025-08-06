import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- USER INPUT ---
# Define bounding box coordinates
lat_min, lat_max = 43.25, 44.25
lon_min, lon_max = -76.25, -75.25

# --- END USER INPUT ---

# Define box corners
lats = [lat_min, lat_max, lat_max, lat_min, lat_min]
lons = [lon_min, lon_min, lon_max, lon_max, lon_min]

# Create plot
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([lon_min - 1, lon_max + 1, lat_min - 1, lat_max + 1])

# Apply cartopy feature to the axis (States, lakes, etc.) using STORMY helper function 
STORMY.add_cartopy_features(ax)

# Add custom formatted gridlines using STORMY function
STORMY.format_gridlines(ax, x_inline=False, y_inline=False, xpadding=20, ypadding=20)

# Plot bounding box
ax.plot(lons, lats, 'b-', linewidth=2, transform=ccrs.PlateCarree(), label='Bounding Box')
ax.scatter([lon_min, lon_max, lon_max, lon_min], 
           [lat_min, lat_min, lat_max, lat_max], 
           color='red', label='Corners', transform=ccrs.PlateCarree())

# Labels and legend
ax.set_title('Bounding Box on Map')
ax.legend()

# Show plot
plt.show()

