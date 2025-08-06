import xarray as xr
from scipy.ndimage import gaussian_filter

# --- USER INPUT ---
# Load the geo_em file
file_path = "geo_em.d01.nc"

# Define Tug Hill region boundaries (adjust if needed)
lat_min, lat_max = 43.25, 44.25
lon_min, lon_max = -76.25, -75.25
flatten_height = 125 # Set new height limit to flatten to (e.g., 100 m)

# --- END USER INPUT ---
ds = xr.open_dataset(file_path)
HGT_attrs = ds.HGT_M.attrs # Extract the attributes for the height variable

# Select grid points inside the region and above 100 m
mask = ((ds.XLAT_M >= lat_min) & (ds.XLAT_M <= lat_max) & 
        (ds.XLONG_M >= lon_min) & (ds.XLONG_M <= lon_max) & 
        (ds.HGT_M > flatten_height))

# Apply flattening
ds["HGT_M"] = ds["HGT_M"].where(~mask, flatten_height)

# Extract the HGT_M variable
hgt = ds["HGT_M"]

# Define the smoothing function (using Gaussian filter for example)
def smooth_terrain(data, sigma):
    """
    Smooth the terrain data using a Gaussian filter.
    `sigma` controls the level of smoothing (higher sigma = more smoothing).
    """
    # Convert to numpy array for smoothing
    smoothed_data = gaussian_filter(data, sigma=sigma)
    return smoothed_data

# Apply smoothing to the entire height data (ignoring the mask for now)
smoothed_hgt = smooth_terrain(hgt.values, sigma=1.5)

# Apply the smoothed data to the regions specified by the mask, leaving the rest unchanged
smoothed_hgt_final = xr.where(mask.values, smoothed_hgt, hgt.values)

# Assign the smoothed terrain back to the original dataset
ds["HGT_M"] = (["Time", "south_north", "west_east"], smoothed_hgt_final)
ds['HGT_M'].attrs.update(HGT_attrs)

# Save the file with the flattened terrain
ds.to_netcdf("geo_em.flattened.nc")
print(f"Flattened Tug Hill region where HGT_M > {flatten_height} m.")
