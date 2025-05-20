import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

'''
Time series of band morphology with day/night shaded
Printed output of morphology in each day/night period
'''

# --- User Input ---
date = "20230319" # YYYYMMDD

# --- End User Input ---
file_path = f"/data2/white/DATA/MET399/morphs/morphs{date}.csv"  
df = pd.read_csv(file_path, sep=r"\s+", header=None, names=["Date", "Hour", "Band"])
df["Datetime"] = pd.to_datetime(df["Date"], format="%m/%d/%Y") + pd.to_timedelta(df["Hour"], unit="h")

# Band label mapping
band_labels = {
    0: "None",
    1: "Broad Coverage",
    2: "LLAP",
    3: "Hybrid",
    4: "Orographic",
}

# Replace this dictionary with accurate sunrise/sunset for each date in UTC, Replace with API in future
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

data_start = df["Datetime"].min()
data_end = df["Datetime"].max()

# Plot
plt.figure(figsize=(14, 5))

# Plot band morphology with time
plt.step(df["Datetime"], df["Band"], linewidth=2, label="Band Type")

# Set xlim based on end datetime
plt.xlim(df["Datetime"].min(), data_end)

# Add y-axis labels
plt.yticks(list(band_labels.keys()), [band_labels[b] for b in band_labels])
plt.xlabel("Time (UTC)")
plt.ylabel("Band Type")
plt.title(f"Lake-Effect Morphologies from {data_start} to {data_end}")

for date_str, (sunrise_str, sunset_str) in sun_times.items():
    sunrise = datetime.strptime(f"{date_str} {sunrise_str}", "%Y-%m-%d %H:%M")
    sunset  = datetime.strptime(f"{date_str} {sunset_str}", "%Y-%m-%d %H:%M")
    
    # Only plot if the sunrise or sunset are within our time period
    if sunrise < data_end or sunset > data_start:
        plt.axvspan(max(sunrise, data_start), min(sunset, data_end), color='gold', alpha=0.2,label="Daylight")

# After all plotting and before plt.show()
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))  # Removes duplicates of sunset/sunrise label

plt.legend(by_label.values(), by_label.keys())

# Formatting
print("min", df["Datetime"])
plt.grid(True, which='both', axis='x', linestyle='--', alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

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

# Apply day/night function to all times
df["Period"] = df.apply(get_day_period, axis=1, sun_times=sun_times)

# Count occurrences of each band type in each period (day/night)
band_period_counts = df.groupby(["Period", "Band"]).size().unstack(fill_value=0)

# Rename columns for readability
band_period_counts.columns = [band_labels.get(b, f"Type {b}") for b in band_period_counts.columns]


print("Band Type Counts by Day/Night:\n")
print(band_period_counts)
