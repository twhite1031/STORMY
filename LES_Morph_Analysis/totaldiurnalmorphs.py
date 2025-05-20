import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
# Your event dates

dates = ["20181110", "20200227", "20220110", "20220123", "20230319","20201102","20221117","20191231","20190305"]
data = []

# Read and combine data from all files
for date in dates:
    file_path = f"/data2/white/DATA/MET399/morphs/morphs{date}.csv"
    df = pd.read_csv(file_path, delim_whitespace=True, header=None, names=["Date", "Hour", "Band"])
    if not pd.api.types.is_numeric_dtype(df["Band"]):
        print(f"Non-numeric values in: {file_path}")
        print(df["Band"].unique())  # Shows what unusual values are present
    # Convert to datetime
    df["Datetime"] = pd.to_datetime(df["Date"], format="%m/%d/%Y") + pd.to_timedelta(df["Hour"], unit="h")
    
    # Append to the combined list
    data.append(df)

# Concatenate all event data
all_data = pd.concat(data, ignore_index=True)

# Define Lake-Effect Band Categories
band_labels = {
    0: "No Lake-Effect",
    1: "Broad Coverage",
    2: "LLAP",
    3: "Hybrid",
    4: "Orographic"
}
# --- Replace this dictionary with accurate sunrise/sunset for each date in UTC ---
sun_times = {
    "2020-11-02": ("11:43",	"21:55"),
    "2020-11-03": ("11:44",	"21:53"),
    "2023-03-18": ("11:13", "23:15"),
    "2023-03-19": ("11:11", "23:16"),
    "2018-11-10": ("11:53", "21:46"),
    "2018-11-11": ("11:54",	"21:45"),
    "2022-11-17": ("12:02",	"21:39"),
    "2022-11-18": ("12:03",	"21:38"),
    "2022-11-19": ("12:04",	"21:37"),
    "2022-11-20": ("12:05",	"21:36"),
    "2019-12-30": ("12:38",	"21:38"),
    "2019-12-31": ("12:38",	"21:39"),
    "2020-01-01": ("12:38",	"21:40"),
    "2022-01-09": ("12:38",	"21:48"),
    "2022-01-10": ("12:37",	"21:49"),
    "2022-01-11": ("12:37",	"21:50"),
    "2022-01-22": ("12:31",	"22:04"),
    "2022-01-23": ("12:30",	"22:05"),
    "2022-01-24": ("12:29",	"22:06"),
    "2020-02-27": ("11:46",	"22:51"),
    "2020-02-28": ("11:44",	"22:53"),
    "2020-02-29": ("11:43",	"22:54"),
    "2019-03-04": ("11:37",	"22:58"),
    "2019-03-05": ("11:36",	"22:59"),
    "2019-03-06": ("11:34",	"23:00"),
    "2019-03-07": ("11:32",	"23:02"),
    "2019-03-08": ("11:30",	"23:03"),
    # Add more entries as needed (format: "YYYY-MM-DD": (sunrise_utc_str, sunset_utc_str))
}

# Seasonality
def get_season(month):
    if month in [10, 11]:  # Oct–Nov
        return "Early"
    elif month in [12, 1]:  # Dec–Jan
        return "Prime"
    elif month in [2, 3, 4]:  # Feb–Apr
        return "Late"
    else:
        return "Off-season"  # Optional catch-all

all_data["Month"] = all_data["Datetime"].dt.month
all_data["Season"] = all_data["Month"].apply(get_season)


# Function to determine if a timestamp is during daylight
def get_day_period(row, sun_times):
    date_str = row["Datetime"].strftime("%Y-%m-%d")
    if date_str in sun_times:
        sunrise_str, sunset_str = sun_times[date_str]
        sunrise = datetime.strptime(f"{date_str} {sunrise_str}", "%Y-%m-%d %H:%M")
        sunset  = datetime.strptime(f"{date_str} {sunset_str}", "%Y-%m-%d %H:%M")
        return "Day" if sunrise <= row["Datetime"] < sunset else "Night"
    else:
        return "Unknown"  # fallback if date isn't in sun_times

# Apply to DataFrame
all_data["Period"] = all_data.apply(get_day_period, axis=1, sun_times=sun_times)
# Count occurrences of each band type in each period
band_period_counts = all_data.groupby(["Period", "Band"]).size().unstack(fill_value=0)
# Rename columns for readability
band_period_counts.columns = [band_labels.get(b, f"Type {b}") for b in band_period_counts.columns]
print("Band Type Counts by Day/Night:\n")
print(band_period_counts)

# Looking at diurnal variations within the seasons
# Group and count
seasonal_counts = all_data.groupby(["Season", "Period", "Band"]).size().unstack(fill_value=0)

# Optional: rename band type columns
seasonal_counts.columns = [band_labels.get(b, f"Type {b}") for b in seasonal_counts.columns]
print("Diurnal Band Type Counts by Season:\n")
print(seasonal_counts)

# Group and count
daynight_counts = all_data.groupby(["Period", "Band"]).size().unstack(fill_value=0)

# Convert Period index to list
periods = daynight_counts.index.tolist()
morphs = daynight_counts.columns.tolist()

# Bar width and positions
bar_width = 0.15
x = np.arange(len(periods))  # [0, 1] for Day and Night
offsets = np.linspace(-bar_width*2, bar_width*2, len(morphs))  # Adjust if you have more morphs

fig, ax = plt.subplots(figsize=(12,6))

for i, morph in enumerate(morphs):
    ax.bar(x + offsets[i], daynight_counts[morph], width=bar_width, label=band_labels[morph])

ax.set_xticks(x)
ax.set_xticklabels(periods)
plt.title("Lake-Effect Morphology Frequency by Time of Day",fontsize=22,fontweight='bold')
plt.xlabel("Time of Day",fontsize=16,fontweight='bold')
plt.ylabel("Hours",fontsize=16,fontweight='bold')
plt.xticks(rotation=0)
plt.legend(title="Morphology")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
