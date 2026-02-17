'''
Plot any Semtinel 2 L2 band on a lat/lon map
'''
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.crs import CRS
import os
resolution = 10  # 10m, 20m, or 60m
band = "BO2" # Input B01 to B12, or "TCI" for true color image
path_to_data = r"C:\Users\thomas.james.white\Documents\SENTINEL_files\S2C_MSIL2A_20250730T171921_R012_T14UPU_20250730T224316\S2C_MSIL2A_20250730T171921_R012_T14UPU_20250730T224316_B02.tif"
#data_selected = rf"R{resolution}m\T17TQJ_20210123T160549_{band}_{resolution}m.jp2"
#band_path = os.path.join(path_to_data, data_selected)

with rasterio.open(path_to_data) as src:
    values = src.read(1).astype("float32")
    profile = src.profile
    transform = src.transform
    crs = src.crs
    band_count = src.count
    if band_count >= 3:
        print("Assuming first 3 bands are RGB")
        values = src.read([1,2,3]).astype("float32")  # Read first 3 bands for RGB
        values = np.transpose(values / values.max(), (1, 2, 0))
    print("CRS:", crs, "Transform:", transform, "Profile:", profile)

# If crs is missing, assume UTM zone 17N (adjust as needed)
if crs is None:
    crs = CRS.from_epsg(32617)  # <-- UTM zone 17N

print(f"Shape of band is: {values.shape}" )

plt.imshow(values)
plt.title(f"Sentinel-2 {band} at {resolution}m resolution")
plt.show()
