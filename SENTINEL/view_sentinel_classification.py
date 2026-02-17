'''
View the classified raster that was created.
Maybe remove cloud edges or merge nearby cloud regions
'''

import rasterio
from rasterio.crs import CRS
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import pandas as pd 
from pyproj import Transformer
from skimage.util import img_as_ubyte
from scipy.ndimage import uniform_filter, binary_erosion
from skimage.measure import label, regionprops
import sys
import os 

# We may adjust this for classified_mlc or classified_mlc_masked
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR,"Sentinel_ClassificationOutputs")
os.makedirs(OUT_DIR,exist_ok=True) # Make directory if missing
OUT_CLASS_TIF = os.path.join(OUT_DIR, "sentinel_classified_mlc.tif")

# Integer ID to class name mapping
CLASS_NAMES = {
    1: 'water',
    2: 'snowcover',
    3: 'cloud',
    4: 'unknown',
    5: 'abnormal',
}
resolution = 20  # Sentinel-2 band resolution to use for classification (10, 20, or 60)
variance_band = 11 # Band used for texture calculation (e.g., B11 for 60m)
variance_filter = 35 # Percentile threshold for high-variance clouds (e.g. 70 = top 30% are considered high variance)
estimated_wind_dir = 320 # degrees

# ----------------------------------------------------------
# Load classified raster
# ----------------------------------------------------------
with rasterio.open(OUT_CLASS_TIF) as src:
    class_img = src.read(1)
    transform = src.transform
    crs = src.crs
    if crs is not None:
        print("CRS:", crs)
    else: 
        crs = CRS.from_epsg(32617)

# Find which class IDs exist
unique = np.unique(class_img)
print("IDs found:", unique)

# Number of classes INCLUDING 0
max_id = int(unique.max())
'''
# ----------------------------------------------------------
# Build dynamic colormap with one color per class ID
# ----------------------------------------------------------
# Use tab20 for up to 20 distinct colors
CLASS_COLORS = {
    "water": (0.1, 0.3, 0.8, 1.0),   # blue
    "snowcover":   (0.5, 1.0, 1.0, 1.0),   # cyan
    "cloud": (1.0, 1.0, 1.0, 1.0),   # white
    "unknown": (0.5, 0.5, 0.5, 1.0),  # gray
    "abnormal" : (1.0, 0.0, 1.0, 1.0)  # magenta
}
colors = []
for cid in range(max_id + 1):
    if cid == 0:
        colors.append((0, 0, 0, 0))  # transparent for ID=0
    else:
        cname = CLASS_NAMES[cid]
        colors.append(CLASS_COLORS[cname])
cmap = ListedColormap(colors)

# ----------------------------------------------------------
# Display classified image
# ----------------------------------------------------------
plt.figure(figsize=(10, 8))
plt.imshow(class_img, cmap=cmap, vmin=0, vmax=max_id)
plt.title("Classified Sentinel 2 Image")
#plt.axis("off")

# ----------------------------------------------------------
# Build legend automatically
# ----------------------------------------------------------
legend_patches = []
for cid in unique:
    if cid == 0 or cid > 3:
        continue  # skip transparent class

    # Use class name if provided, otherwise generic label
    legend_label = CLASS_NAMES.get(cid, f"Class {cid}")

    patch = Patch(facecolor=cmap(cid), label=f"{cid}: {legend_label}")
    legend_patches.append(patch)

plt.legend(handles=legend_patches, loc="lower right",
           title="Classes", framealpha=0.9)

plt.show()
'''
# Build row/col index grid
H, W = class_img.shape
rows, cols = np.indices((H, W))

df = pd.DataFrame({
    "class_id": class_img.ravel(),
    "row": rows.ravel(),
    "col": cols.ravel(),
})
'''
# Extract all pixel locations + class IDs to store later
records = []

for r, c in zip(rows.ravel(), cols.ravel()):
    # Get class ID at this pixel
    cid = int(class_img[r, c])
    records.append({"class_id": cid, "row": r, "col": c})

df = pd.DataFrame(records)
'''
print("Total extracted pixels:", len(df))
print(df["class_id"].value_counts())
H, W = class_img.shape

# Identifying spatial features of the cloud class
cloud_id = 3 # example, change to match your MLC class number

# Cloud mask
cloud_mask = (class_img == cloud_id)
print("Cloud pixels:", cloud_mask.sum()) 

# -----------------------------------------------------------
# Load reflectance band (do NOT mask yet)
# -----------------------------------------------------------
band_path = fr"d:\S2B_MSIL2A_20210123T160549_N0500_R097_T17TQJ_20230522T161321.SAFE\S2B_MSIL2A_20210123T160549_N0500_R097_T17TQJ_20230522T161321.SAFE\GRANULE\L2A_T17TQJ_A020286_20210123T160552\IMG_DATA\R{resolution}m\T17TQJ_20210123T160549_B{variance_band}_{resolution}m.jp2" 
with rasterio.open(band_path) as src:
    reflect_img = src.read(1).astype("float32")
    print("Reflectance band shape:", reflect_img.shape)

# -----------------------------------------------------------
# Compute texture
# -----------------------------------------------------------
# Convert to 8-bit for texture calculation
img8 = img_as_ubyte((reflect_img - reflect_img.min()) /
                    (reflect_img.max() - reflect_img.min()))

window = 7 # 3x3 pixel window to capture patterns

# Slide a window and compute local variance
# mean and mean_sq are simply the sum and sum of squares in the window (no need to divide by N again)
mean = uniform_filter(img8.astype(float), size=window)
mean_sq = uniform_filter((img8.astype(float)**2), size=window)
variance = mean_sq - mean**2

# Shrinks the cloud mask itself to avoid edge effects
#core_cloud_mask = binary_erosion(cloud_mask, iterations=2) # Remove edge effects since we calculated mean on full image

# Apply cloud mask AFTER texture calculation
variance_cloud = np.where(cloud_mask, variance, np.nan)

print("Variance texture computed.")
'''
#plt.imshow(variance_cloud, cmap="viridis")
plt.imshow(variance_cloud, cmap='Reds', vmin=0, vmax=np.nanpercentile(variance_cloud, 99))

plt.colorbar(label="Local Variance")
plt.title("Cloud Texture Variance")
plt.show()
'''
# Split into smooth / rough clouds based on variance threshold
cloud_var_values = variance_cloud[~np.isnan(variance_cloud)] # Non-nan values

# e.g., top __% most textured = "high variance"
thr = np.percentile(cloud_var_values, variance_filter)
print(f"Variance threshold for high-variance clouds: {thr:.2f} (top {100 - variance_filter}%)")
high_var_mask = (variance_cloud >= thr)   # textured clouds
low_var_mask  = (variance_cloud <  thr)   # smooth clouds

# Within cloud only and valid threshold comparison
high_var_mask &= np.isfinite(variance_cloud)
low_var_mask  &= np.isfinite(variance_cloud)

# 0 = non-cloud, 1 = stratiform, 2 = cellular, 3 = band/roll
cloud_type = np.zeros_like(class_img, dtype=np.uint8)

# Stratiform: low variance clouds
#cloud_type[low_var_mask] = 1

# Optional features to add in future
pixel_size = 20  # meters per pixel 

min_band_length_m = 1500  # require 1.5 km long
min_band_length_px = min_band_length_m / pixel_size

# Label connected cloud regions
labels = label(cloud_mask, connectivity=2)
props = regionprops(labels)

# Get the area of each region
areas = [p.area for p in props]

print("Number of cloud regions:", len(areas))
print("Mean region area (pixels):", np.mean(areas))
print("Median region area (pixels):", np.median(areas))
print("Min region area:", np.min(areas))
print("Max region area:", np.max(areas))
# Try to get an angle of the wind direction from estimated_wind_dir
wind_to_deg = ((estimated_wind_dir) + 180.0) % 360.0
math_deg = 90.0 - wind_to_deg
theta_band = np.deg2rad(math_deg)
def angle_diff(a, b):
    d = np.abs(a - b)
    return np.minimum(d, np.pi - d)  # smallest angle, modulo 180°

for region in props:
    area = region.area # Number of pixels
    maj  = region.major_axis_length
    minr = region.minor_axis_length
    sol = region.solidity
    theta = region.orientation
    delta_theta = angle_diff(theta, theta_band)
    ddeg = delta_theta * (180.0 / np.pi)
    if minr == 0:
        continue
    aspect = maj / minr

    # Get pixels for this region
    rr, cc = region.coords[:, 0], region.coords[:, 1]

    # Get the variance of the region
    variance_region = variance_cloud[rr, cc].mean()
    
    # Heuristic rules to tune cloud type
    # Remove small noise
    if area < 40:
        continue

    # LLAP plumes: large and elongated
    elif area < 100000 and variance_region >= thr:
        cloud_type[rr, cc] = 2
    else:
        cloud_type[rr, cc] = 1
    # Cellular convection: roundish + high variance
    #elif variance_region > thr and aspect <= 2.5:
    #    cloud_type[rr, cc] = 2

 

    #print(f"Region {region.label}: area={area}, maj={maj:.1f}, min={minr:.1f}, aspect={aspect:.2f}, mean_var={variance_region:.1f}, ddeg = {ddeg} -> class {cloud_type[rr[0], cc[0]]}")
    if cloud_type[rr[0], cc[0]] == 3:
        print("degrees from wind direction:", ddeg)

# Viusalize output
morph_cmap = ListedColormap([
    (0, 0, 0, 0),      # 0 noiss, transparent
    (0.6, 0.6, 0.9, 1),# 1 LLAP, purple
    (0.2, 0.8, 0.2, 1),# 2 cellular, bright green
])

plt.figure(figsize=(10,8))
plt.imshow(cloud_type, cmap=morph_cmap, vmin=0, vmax=3)
plt.title("Cloud Morphology Classification")

# Create custom legend
legend_elements = [
    Patch(facecolor=(0.6, 0.6, 0.9, 1), edgecolor='k', label='Uniform Band'),
    Patch(facecolor=(0.2, 0.8, 0.2, 1), edgecolor='k', label='Developing / Cellular'),
    Patch(facecolor=(0, 0, 0, 0), edgecolor='k', label='Noise / Unclassified')
]

plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.show()
    
sys.exit()
# Prepare DataFrame per class and convert coords
class_pixels = {}

for cid in unique:
    if cid == 0:  # skip nodata
        continue

    px = df[df["class_id"] == cid].copy()

    # Check to see if any pixels exist within the class
    if len(px) == 0:
        continue

    # Convert pixel → UTM meters
    utm_x, utm_y = rasterio.transform.xy(transform, px["row"].values, px["col"].values)

    # Store the utm coords
    px["x_utm_m"] = utm_x
    px["y_utm_m"] = utm_y

    # Convert UTM → lat/lon 
    to_geo = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = to_geo.transform(px["x_utm_m"].values, px["y_utm_m"].values)
    px["lon_deg"] = lon
    px["lat_deg"] = lat

    class_pixels[cid] = px
    print(f"Extracted {len(px)} pixels for {cid}: {CLASS_NAMES[cid]}")

    # --- SAVE CSV PER CLASS HERE  ---
    # We can see where the classes appeared most frequently geographically
    out_csv = os.path.join(os.path.dirname(OUT_CLASS_TIF), f"class_{cid}_{CLASS_NAMES[cid]}.csv")
    px[["class_id", "row", "col", "x_utm_m", "y_utm_m", "lon_deg", "lat_deg"]].to_csv(out_csv, index=False)
    print("Saved CSV:", out_csv)