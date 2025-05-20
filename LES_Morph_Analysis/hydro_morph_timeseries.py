import os
import warnings
import numpy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pyart
from pyart.testing import get_test_data
from cartopy.feature import NaturalEarthFeature
import cartopy.feature as cfeature
from metpy.plots import USCOUNTIES
import radarfuncs
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import pandas as pd
warnings.filterwarnings('ignore')

'''
Makes a timeseries plot with both hydrometeor frequency and
morphologies
'''

# === USER INPUTS ===
date = "20220123"
step = timedelta(minutes=5) # Timestep to loop through
radar = 'KTYX'
var = 'N0H' # Level III radar variable

radar_data_dir = f"/data2/white/DATA/MET399/NEXRADLVL3/{date}/"
savepath = f"/data2/white/PLOTS_FIGURES/MET399/{date}/"
# --- End User Input ---

file_path = f"/data2/white/DATA/MET399/morphs/morphs{date}.csv" 


time_list = []
class_counts_list = []

# Read in Morphology data
df = pd.read_csv(file_path, sep=r"\s+", header=None, names=["Date", "Hour", "Band"])
df["Datetime"] = pd.to_datetime(df["Date"], format="%m/%d/%Y") + pd.to_timedelta(df["Hour"], unit="h")
start_time = df["Datetime"].min()
end_time = df["Datetime"].max()

# Band label mapping
band_labels = {
    0: "None",
    1: "Broad Coverage",
    2: "LLAP",
    3: "Hybrid",
    4: "Orographic",
}

# === LOOP THROUGH TIME ===
current_time = start_time
while current_time <= end_time:
    radar_file = radarfuncs.find_closest_radar_lvl3_file(radar_data_dir, var, radar, current_time)

    if radar_file:
        
        # Read in Level III data
        src = os.path.join(radar_data_dir, radar_file)
        radar_object = pyart.io.read_nexrad_level3(src)
        hhc = radar_object.fields['radar_echo_classification']['data']
        filename = radar_file 
            
        # Slice out date and time from the last 12 characters
        datetime_part = filename[-12:]  # '201811100707'

        # Convert to datetime object
        dt = datetime.strptime(datetime_part, "%Y%m%d%H%M")

        # Format to desired output string
        filename_str = dt.strftime("%b %d_%H%M")

        # Mask out class 0
        hhc_filtered = hhc[hhc >= 10]

        # Hydrometeor Classification labels
        labels = ["ND", "BI", "GC", "IC", "DS", "WS", "RA", "HR", "BD", "GR", "HA", "LH", "GH", "--", "UK", "RH"]
        ticks = np.arange(5, 156, 10)  # Hydrometeor bin centers for plotting

        boundaries = np.arange(0, 161, 10) # Hydrometoer Classification boundaries
        counts, bin_edges = numpy.histogram(hhc_filtered,bins=boundaries)

        time_list.append(dt)
        class_counts_list.append(counts)
        
    else:
        print(f"No file found for {current_time}")
    current_time += step


class_counts_array = np.array(class_counts_list)  # shape: [time, class]
times = np.array(time_list)

# --- To normalize hydrometeors ---
# Calculate total valid pixels per time step
totals = class_counts_array.sum(axis=1).reshape(-1, 1)

# Avoid division by zero
totals[totals == 0] = 1

# Normalize: divide each row (time step) by the total count
normalized_array = class_counts_array / totals

# Specific Labels to use
selected_labels = ["ND", "BI", "GC", "IC", "DS", "WS", "RA", "HR", "BD", "GR", "HA", "LH", "GH", "--", "UK", "RH"]

# Looks up the correct index for the labels provided (Matches color key)
selected_indices = [labels.index(label) for label in selected_labels]

class_colors = {
    "ND":  "#C0C0C0",  # Light gray for "No Data"
    "BI":  "#FF69B4",  # Hot pink for Biological
    "GC":  "#FFD700",  # Gold for Ground Clutter
    "IC":  "#00FFFF",  # Cyan for Ice Crystals
    "DS":  "#ADD8E6",  # Light Blue for Dry Snow
    "WS":  "#9370DB",  # Medium purple for Wet Snow
    "RA":  "#00FF00",  # Bright green for Rain
    "HR":  "#007700",  # Darker green for Heavy Rain
    "BD":  "#FFA500",  # Orange for Big Drops
    "GR":  "#FF6347",  # Tomato for Graupel
    "HA":  "#8B0000",  # Dark red for Hail
    "LH":  "#FF0000",  # Bright red for Large Hail
    "GH":  "#B22222",  # Firebrick for Giant Hail
    "--":  "#000000",  # Black or ignored
    "UK":  "#A9A9A9",  # Dark gray for Unknown
    "RH":  "#0000FF",  # Blue for Refrozen Hydrometeors
}

# --- Average Frequency Overall ---
average_freq = normalized_array.mean(axis=0) * 100

print(" Overall Average Hydrometeor Frequencies (%):")
for i, label in enumerate(labels):
    print(f"{label}: {average_freq[i]:.2f}%")

# ---- Plotting section for line plot ----

fig, ax1 = plt.subplots(figsize=(14, 5))

# Plot selected hydrometeor classes
for i in selected_indices:
    label = labels[i]
    color = class_colors.get(label, None)
    #ax1.plot(times, class_counts_array[:, i], label=label, color=color)
    ax1.plot(times, normalized_array[:, i] * 100, label=label, color=color)

# Add band morphology on secondary axis
ax2 = ax1.twinx()
ax2.step(df["Datetime"], df["Band"], where='post', linewidth=2, color="black", label="Band Type")
ax2.set_ylabel("Band Morphology")
ax2.set_yticks(list(band_labels.keys()))
ax2.set_yticklabels([band_labels[b] for b in band_labels])

# Format and legend
ax1.set_xlabel("Time")
ax1.set_ylabel("Hydrometeor Frequency (%)")
ax1.legend(title="Hydrometeor Class", bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.3)

# Format x-axis
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d %H:%M'))
ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(rotation=45)
plt.tight_layout()
fig.suptitle("Hydrometeor Class Frequencies Over Time", y=1.02)

# Save + show
plt.savefig(savepath + f"hydro_morph_timeseries{date}.png", dpi=300)
plt.show()



