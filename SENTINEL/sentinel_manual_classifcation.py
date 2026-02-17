'''
Classify any classes using Maximum Likelihood Classifier (QDA) on Sentinel 2
 data by selecting training polygons manually.
'''
import numpy as np
import rasterio
from rasterio.crs import CRS
import sys
from shapely.geometry import Polygon
from matplotlib.path import Path as MplPath
from pyproj import Transformer
import pandas as pd
import os
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    cohen_kappa_score,
    classification_report,
)
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy.ma as ma
import xml.etree.ElementTree as ET
import re

# Set Paths
# Folder containing Sentinel-2 L2 GRANULE folder
DATA_DIR = r"D:\S2B_MSIL2A_20210123T160549_N0500_R097_T17TQJ_20230522T161321.SAFE\S2B_MSIL2A_20210123T160549_N0500_R097_T17TQJ_20230522T161321.SAFE"

# CSVs containing training and validation points
# Each CSV must have columns: row,col,class_id
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TV_DIR = os.path.join(BASE_DIR,"Sentinel_TrainingValidation")
OUT_DIR = os.path.join(BASE_DIR,"Sentinel_ClassificationOutputs")
os.makedirs(TV_DIR,exist_ok=True) # Make directory if missing
os.makedirs(OUT_DIR,exist_ok=True) # Make directory if missing
TRAIN_CSV = os.path.join(BASE_DIR, "Sentinel_TrainingValidation", "sentinel_training_pixels.csv")
VAL_CSV   = os.path.join(BASE_DIR, "Sentinel_TrainingValidation", "sentinel_validation_pixels.csv")

# Output classified raster
OUT_CLASS_TIF = os.path.join(OUT_DIR, "sentinel_classified_mlc.tif")
# Path to metadata file
MTL_XML = r"D:\S2B_MSIL2A_20210123T160549_N0500_R097_T17TQJ_20230522T161321.SAFE\S2B_MSIL2A_20210123T160549_N0500_R097_T17TQJ_20230522T161321.SAFE\MTD_MSIL2A.xml"

find_classes = False # Set to False if you already have the regions of polygons
resolution = 20  # Sentinel-2 band resolution to use for classification (10, 20, or 60)
# Reference band for georeferencing (any band is fine)
ref_band_path =  fr"d:\S2B_MSIL2A_20210123T160549_N0500_R097_T17TQJ_20230522T161321.SAFE\S2B_MSIL2A_20210123T160549_N0500_R097_T17TQJ_20230522T161321.SAFE\GRANULE\L2A_T17TQJ_A020286_20210123T160552\IMG_DATA\R{resolution}m\T17TQJ_20210123T160549_B03_{resolution}m.jp2" 

# Bands you'd like to use for classification
SR_BANDS = [2, 3, 4]
TR_BANDS = [11,12]
MAX_SAMPLES_PER_CLASS = 500 # Max samples per class to avoid imbalance for training/validation

# Map class names to integer IDs
CLASS_NAMES = {
    'water': 1,
    'snowcover': 2,
    'cloud': 3,
}

#class_polygons = None
class_polygons = {
    "water": Polygon([
        (-77.8, 43.95), # top-right
        (-78, 43.95), # top-left
        (-78, 43.86), # bottom-left
        (-77.8, 43.86), # bottom-right
        ]),
    "snowcover": Polygon([
        (-78.2, 44.1), 
        (-78.3, 44.1), 
        (-78.3, 44.04), 
        (-78.2, 44.04), 
    ]),
    'cloud': Polygon([
        (-77.8, 43.36),
        (-77.9, 43.36),
        (-77.9, 43.3),
        (-77.8, 43.3),
    ]),
}

def view_rgb_image(bands_by_res):
    """Helper to quickly view an RGB image array."""
    # Use 10m true color image if available
    with rasterio.open(bands_by_res[10]["TCI"]) as src:
        values = src.read([1,2,3]).astype("float32")  # Read first 3 bands for RGB
        values = np.transpose(values / values.max(), (1, 2, 0))
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
    plt.figure(figsize=(10,10))
    plt.imshow(values)
    plt.axis("off")
    plt.show()

def create_lat_lon():
    ref_band_path =  r"d:\S2B_MSIL2A_20210123T160549_N0500_R097_T17TQJ_20230522T161321.SAFE\S2B_MSIL2A_20210123T160549_N0500_R097_T17TQJ_20230522T161321.SAFE\GRANULE\L2A_T17TQJ_A020286_20210123T160552\IMG_DATA\R10m\T17TQJ_20210123T160549_B02_10m.jp2"
    with rasterio.open(ref_band_path) as src:
        img = src.read(1)  # read first band (or stack RGB if needed)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    # Sentinel doesn't have crs in data, so we assume UTM zone 17N if missing
    if crs is None:
        crs = CRS.from_epsg(32617)  # UTM zone 17N

    # 2. Compute real coordinates for each pixel in dataset CRS
    rows, cols = np.indices((img.shape[0], img.shape[1]))
    xs, ys = rasterio.transform.xy(transform, rows, cols)

    # These arrays are UTM exact positions
    x_map = np.array(xs)
    y_map = np.array(ys)

    # If needed, reproject to geographic (EPSG:4326)
    # PlateCarre requires lon/lat
    if not crs.is_geographic:
        from pyproj import Transformer
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon_map, lat_map = transformer.transform(x_map, y_map)
    else:
        print("CRS is already geographic.")
        lon_map, lat_map = x_map, y_map  # already lon/lat
    return lon_map, lat_map
lon_map, lat_map = create_lat_lon()

def view_rgb_on_map(rgb,lon_map, lat_map, test_polygons=None):

     # Use 10m true color image if available
    with rasterio.open(os.path.join(DATA_DIR,bands_by_res[10]["TCI"])) as src:
        values = src.read([1,2,3]).astype("float32")  # Read first 3 bands for RGB
        values = np.transpose(values / values.max(), (1, 2, 0))
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
    plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_title("WGS 84 Projection")

    # 5. Display image using georeferenced extent
    # Cartopy expects (xmin, xmax, ymin, ymax)
    ax.imshow(
        values,
        origin="upper",
        extent=[lon_map.min(), lon_map.max(), lat_map.min(), lat_map.max()],
        transform=ccrs.PlateCarree()
    )

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, linestyle="--")  # thin lines
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 12}  # smaller font
    gl.ylabel_style = {'size': 12}
    plt.tick_params(width=0.5, labelsize=8)
    ax.set_frame_on(False)
    plt.axis('off')

    # ---------- ADD POLYGONS + LEGEND ----------
    if test_polygons:
        legend_handles = []  # store Line2D items for legend

        colors = {
            "snowcover": "yellow",
            "water": "blue",
            "land": "green",
            "cloud": "green"
        }

        for name, poly in test_polygons.items():
            x, y = poly.exterior.xy
            color = colors.get(name, "yellow")  # fallback color

            # ax.plot returns a list — take the first Line2D object
            line_obj = ax.plot(
                x, y, color=color, linewidth=2,
                transform=ccrs.PlateCarree(), label=name
            )[0]

            legend_handles.append(line_obj)

        # Create legend outside the map
        ax.legend(
            handles=legend_handles,
            title="Training Polygons",
            loc="lower left",
            framealpha=0.8
        )

    
    plt.show()

tree = ET.parse(MTL_XML)
root = tree.getroot()

# Try namespace-aware search:
ns = {"n1": "https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd"}

scale_node = root.find(".//n1:BOA_QUANTIFICATION_VALUE", ns)

# If not found, fallback: wildcard namespace
if scale_node is None:
    scale_node = root.find(".//{*}BOA_QUANTIFICATION_VALUE")

scale = float(scale_node.text)
print("S2 BOA scale factor =", scale)

# --- Offsets ---
offset_nodes = root.findall(".//{*}BOA_ADD_OFFSET")

# Store for future usage 
# ESA-defined band order for L2A offsets:
offset_band_order = [
    "B01", "B02", "B03", "B04",
    "B05", "B06", "B07", "B8A",
    "B08", "B09", "B11", "B12", "B10"
]

offsets = {}

for idx, node in enumerate(offset_nodes):
    band_name = offset_band_order[idx]
    offsets[band_name] = float(node.text)

print("Offsets by band:", offsets)

# Grab all filesnames for all bands in each resolution
# Namespace wildcard needed for ESA XML
image_files = root.findall(".//{*}IMAGE_FILE")

# Storage:
bands_by_res = {}  # plain dict

band_regex = re.compile(r"_(B\d{1,2}[A]?)_")   # captures B02, B8A, B11, etc.
other_regex = re.compile(r"_(AOT|WVP|SCL|TCI)_")  # other jp2 layers

for node in image_files:
    fname = node.text.strip()

    # Determine resolution from path
    if "/R10m/" in fname:
        res = 10
    elif "/R20m/" in fname:
        res = 20
    elif "/R60m/" in fname:
        res = 60
    else:
        continue

    if res not in bands_by_res:
        bands_by_res[res] = {}  # create the inner dict

    # Extract band ID (B02, B8A, etc.)
    m = band_regex.search(fname)

    # Check if file is spectral band, otherwise check if its an ancillary layer such as Water vapor (WVP) or Aerosol Optical Thickness (AOT)
    if m:
        band_id = m.group(1)  # e.g., 'B02' or 'B8A'
    else:
        # Try ancillary layer match
        m2 = other_regex.search(fname)
        if m2:
            band_id = m2.group(1)  # e.g., 'AOT', 'WVP', 'SCL', 'TCI'
        else:
            continue  # skip unknown files

    # Store full path of all bands
    filetype = ".jp2" 
    bands_by_res[res][band_id] = fname + filetype

# ---- Print results ----
for res, bands in bands_by_res.items():
    print(f"\nResolution {res}m bands:")
    for b, f in bands.items():
        print(f"  Band {b}: {f}")

band_paths = {} # Full paths to each band that will be used are stored here 

# band ID normalization since sentinel uses interesting formats like B8A
def normalize_band_id(b):
    """
    Normalize band identifiers to Sentinel-2 canonical form:
      2     -> 'B02'
      '02'  -> 'B02'
      'B02' -> 'B02'
      '8A'  -> 'B8A'
      'B8A' -> 'B8A'
      'AOT' -> 'AOT'
      'SCL' -> 'SCL'
    """
    b = str(b).upper().strip()

    # Ancillary layers (leave unchanged)
    if b in ("AOT", "WVP", "SCL", "TCI"):
        return b

    # Strip only if spectral band was written like B02 or B8A
    if b.startswith("B"):
        b = b[1:]

    # Now b is either numeric (e.g. "2") or alphanumeric ("8A")
    if b.isdigit():
        return "B" + b.zfill(2)

    return "B" + b

try:
    selected_bands_in_resolution = bands_by_res[resolution]
    print("Retrieving selected bands for resolution: ", resolution)
    for b in SR_BANDS + TR_BANDS:
        bid = normalize_band_id(b) # Ensure consistent band ID format

        # Check if available
        if bid not in selected_bands_in_resolution:
            print(f"⚠️ Band {b} not found for resolution {resolution}m")
            continue

        # Build full path
        rel_path = selected_bands_in_resolution[bid] 
        full_path = os.path.join(DATA_DIR, rel_path)

        # Store the band path
        band_paths[bid] = full_path

except KeyError:
    print(f"❌ No bands found for resolution {resolution}m")

# Now we can scale the bands we know we'll use
scaled_bands = {}
raw_first_band = None  # for nodata mask

i = 0 # initialize index for profile saving
for b, path in band_paths.items():
    
    print(f"Reading band {b} from {os.path.basename(path)}")

    with rasterio.open(path) as src:
        dn = src.read(1).astype("float32")
        counts = src.count
        print(f" Band shape: {dn.shape}")
        print(f"Found {counts} band(s) in file.")
              
        # Save profile from the second band (any band is fine):
        if i == 0:
            profile = src.profile
            raw_first_band = dn.copy()
            i += 1

        # Sentinel-2 L2A reflectance scaling:
        # Apply offset 
        corrected = dn + offsets[b]   # offsets[b] came from <BOA_ADD_OFFSET>

        # Divide by BOA_QUANTIFICATION_VALUE
        scaled = corrected / scale

        scaled_bands[b] = scaled

print("Selected band paths:")
for k, v in band_paths.items():
    print(f"  B{k}: {v}")

# Add NDSI if B03 and B11 exis

if "B03" in scaled_bands and "B11" in scaled_bands:
    B03 = scaled_bands["B03"]
    B11 = scaled_bands["B11"]

    eps = 1e-6
    NDSI = (B03 - B11) / (B03 + B11 + eps)
else:
    NDSI = None


# Stack into shape: (bands, rows, cols), easy to grab pixel spectra
band_keys = sorted(scaled_bands.keys())    # ensures stable order
scaled_stack = np.stack([scaled_bands[b] for b in band_keys], axis=0)

# Add NDSI (Normalized Difference Snow Index) as an additional band if computed
# If NDSI exists, append it

if NDSI is not None:
    scaled_stack = np.vstack([scaled_stack, NDSI[np.newaxis, ...]])

print("Final feature stack shape:", scaled_stack.shape)
#print("Band order used:", band_keys + (["NDSI"] if NDSI is not None else []))

n_bands, n_rows, n_cols = scaled_stack.shape
print(f"Stack shape: bands={n_bands}, rows={n_rows}, cols={n_cols}")

# Create a simple nodata mask: here assume DN == 0 in the first band is nodata
nodata_mask = (raw_first_band == 0.0)

# =========================
# 3. QUICK RGB PLOT (VISUAL CHECK)
# =========================
# Apply linear normalization as well as remove extreme 2% pixels
def normalize(arr):
    a_min, a_max = np.nanpercentile(arr, [2, 98])
    arr = np.clip(arr, a_min, a_max)
    return (arr - a_min) / (a_max - a_min + 1e-6)

# Find indices of bands 4,3,2 in SR_BANDS
idx_b4 = SR_BANDS.index(4)
idx_b3 = SR_BANDS.index(3)
idx_b2 = SR_BANDS.index(2)

# Create a (rows, cols, 3) RGB array
rgb = np.dstack([
    normalize(scaled_stack[idx_b4]),
    normalize(scaled_stack[idx_b3]),
    normalize(scaled_stack[idx_b2]),
])

if find_classes:
    view_rgb_on_map(bands_by_res,lon_map,lat_map,test_polygons=class_polygons)
    sys.exit()

# ===========================================================
# 2. Open raster & prepare transforms
# ===========================================================

with rasterio.open(ref_band_path) as src:
    img = src.read(1)
    H, W = src.height, src.width
    crs = src.crs
    if crs is None:
        crs = CRS.from_epsg(32617)  # UTM zone 17N
    transform = src.transform

# Convert lon/lat → raster CRS (projected meters)
transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

# ===========================================================
# 3. Generate full pixel grid coordinate mesh (PROJECTED meters)
# ===========================================================
rows, cols = np.mgrid[0:H, 0:W]
xs, ys = rasterio.transform.xy(transform, rows, cols)
X = np.array(xs)
Y = np.array(ys)

# ===========================================================
# 4. Loop through polygons & build class masks
# ===========================================================

np.random.seed(42) # Consistent randomness
data_train = []
data_val = []

for class_name, poly in class_polygons.items():

    # --- Reproject polygon to match raster CRS ---
    utm_coords = [transformer.transform(x, y) for x, y in poly.exterior.coords]
    poly_proj = Polygon(utm_coords)

    # --- Convert to pixel interior test path ---
    rowcol = [src.index(x, y) for x, y in poly_proj.exterior.coords] # Converts polygon boundaries to pixel indices boundaries
    poly_path = MplPath([(c, r) for r, c in rowcol]) # Create a matplotlib path for point-in-polygon testing

    # --- FAST mask creation (vectorized point-in-polygon test) ---
    pts = np.column_stack((cols.ravel(), rows.ravel())) # (x=col, y=row) points for each pixel in the orignal image
    inside = poly_path.contains_points(pts).reshape(H, W) # Check which pts are inside the polygon (poly_path), reshaping back into original image shape

     # --- Collect valid labeled pixels ---
    pixels = np.argwhere(inside)

    if len(pixels) == 0:
        print("No pixels found for class when splitting:", class_name)
        continue

    # ---- LIMIT PIXELS PER CLASS ----
    if len(pixels) > MAX_SAMPLES_PER_CLASS:
        pixels = pixels[np.random.choice(len(pixels), MAX_SAMPLES_PER_CLASS, replace=False)]

    # --- Split exactly 80/20 per class ---
    idx = np.random.permutation(len(pixels))
    pixels = pixels[idx]
    split = int(0.8 * len(pixels))

    # --- Assign to train/val ---
    for r, c in pixels[:split]:
        if not nodata_mask[r, c]:
            data_train.append({"class_id": class_name, "row": int(r), "col": int(c)})

    for r, c in pixels[split:]:
        if not nodata_mask[r, c]:
            data_val.append({"class_id": class_name, "row": int(r), "col": int(c)})

# Save separately
pd.DataFrame(data_train).to_csv(TRAIN_CSV, index=False)
pd.DataFrame(data_val).to_csv(VAL_CSV, index=False)

print("Saved training & validation CSVs.")

# =========================
# 4. LOAD TRAINING & VALIDATION POINTS
# =========================

def extract_samples_from_csv(csv_path, stack, nodata_mask):
    """
    csv_path: path to CSV with columns row,col,class_id
    stack:    array (bands, rows, cols) of reflectance
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    n_bands, n_rows, n_cols = stack.shape
    X_list = []
    y_list = []

    for _, row in df.iterrows():
        r = int(row["row"])
        c = int(row["col"])
        cls = CLASS_NAMES[row["class_id"].lower()]
        
        # Skip points that fall in nodata or outside image
        if r < 0 or r >= n_rows or c < 0 or c >= n_cols:
            continue
        if nodata_mask[r, c]:
            continue

        spectrum = stack[:, r, c]
        X_list.append(spectrum)
        y_list.append(cls)

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=int)

    print(f"Valid samples after nodata check: {X.shape[0]}")
    return X, y

X_train, y_train = extract_samples_from_csv(TRAIN_CSV, scaled_stack, nodata_mask)
X_val, y_val     = extract_samples_from_csv(VAL_CSV,   scaled_stack, nodata_mask)

# =========================
# 5. TRAIN MLC (QUADRATIC DISCRIMINANT)
# =========================

print("\nTraining Maximum Likelihood (QDA) classifier...")
mlc = QuadraticDiscriminantAnalysis(reg_param=0.01,store_covariance=True)
mlc.fit(X_train, y_train)
print("Training completed.")

# =========================
# 6. CLASSIFY FULL IMAGE
# =========================

# Flatten stack: (bands, rows, cols) → (rows*cols, bands)
X_all = scaled_stack.reshape(n_bands, -1).T  # shape (N_pixels, N_bands)
print("Flattened image for classification:", X_all.shape)

# Predict class for all pixels
print("Classifying full scene (this may take a bit)...")
y_pred_flat = mlc.predict(X_all)

# Reshape back to image
class_map = y_pred_flat.reshape(n_rows, n_cols)

# Apply nodata mask (set to 0 where nodata)
class_map[nodata_mask] = 0  # 0 reserved for nodata

# =========================
# 7. EVALUATE ACCURACY ON VALIDATION SET
# =========================

# Extract predicted class at validation locations
def extract_predicted_labels(csv_path, class_map, nodata_mask):
    df = pd.read_csv(csv_path)

    y_true = []
    y_pred = []

    for _, row in df.iterrows():
        r = int(row["row"])
        c = int(row["col"])
        cls = CLASS_NAMES[row["class_id"].lower()]

        if r < 0 or r >= n_rows or c < 0 or c >= n_cols:
            continue
        if nodata_mask[r, c]:
            continue

        pred_cls = int(class_map[r, c])
        # Optionally skip if classifier predicted nodata:
        if pred_cls == 0:
            continue

        y_true.append(cls)
        y_pred.append(pred_cls)

    return np.array(y_true, dtype=int), np.array(y_pred, dtype=int)

y_true, y_pred = extract_predicted_labels(VAL_CSV, class_map, nodata_mask)

print(f"\nValidation samples used for accuracy: {len(y_true)}")

cm = confusion_matrix(y_true, y_pred, labels=sorted(set(y_true)))
oa = accuracy_score(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)

print("\nConfusion Matrix (rows=true, cols=pred):")
print(cm)

print(f"\nOverall Accuracy: {oa:.3f}")
print(f"Kappa: {kappa:.3f}")

print("\nPer-class report:")
print(classification_report(y_true, y_pred, digits=3))

# Optional: print with class names
print("\nClass ID → Name mapping:")
for cid in sorted(CLASS_NAMES.keys()):
    print(f"  {cid}: {CLASS_NAMES[cid]}")

# =========================
# 8. SAVE CLASSIFIED RASTER
# =========================

out_profile = profile.copy()
out_profile.update(
    dtype=rasterio.uint16,
    count=1,
    nodata=0
)

with rasterio.open(OUT_CLASS_TIF, "w", **out_profile) as dst:
    dst.write(class_map.astype(rasterio.uint16), 1)

print(f"\nSaved classified map to: {OUT_CLASS_TIF}")
print("Done.")