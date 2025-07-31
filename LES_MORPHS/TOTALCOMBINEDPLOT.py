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
from scipy.stats import levene, kruskal, mannwhitneyu
import seaborn as sns
import scikit_posthocs as sp
# === User Input ===
radar = 'KTYX'
var = 'N0H' # Level III data

# Event dates
dates = ["20181110", "20200227", "20220110", "20220123", "20230319","20201102","20221117","20191231","20190305"]
data_savepath = f"/data2/white/DATA/MET399/" # For complete data (csv) file

# --- End User Input

data = []
time_ranges = []
all_summary_data = []
labels = ["ND", "BI", "GC", "IC", "DS", "WS", "RA", "HR", "BD", "GR", "HA", "LH", "GH", "--", "UK", "RH"]
boundaries = np.arange(0, 161, 10)
morph_labels = {
    0: "No Lake-Effect",
    1: "Broad Coverage",
    2: "LLAP",
    3: "Hybrid",
    4: "Orographic",
     }

def get_season(month):
    if month in [10, 11]:  # Oct–Nov
        return "Early"
    elif month in [12, 1]:  # Dec–Jan
        return "Prime"
    elif month in [2, 3, 4]:  # Feb–Apr
        return "Late"
    else:
        return "Off-season"  # Optional catch-all

# Read and combine data from all files
for date in dates:
    file_path = f"/data2/white/DATA/MET399/morphs/morphs{date}.csv"
    radar_data_dir = f"/data2/white/DATA/MET399/NEXRADLVL3/{date}/"
    savepath = f"/data2/white/PLOTS_FIGURES/MET399/{date}/"
    df = pd.read_csv(file_path, delim_whitespace=True, header=None, names=["Date", "Hour", "Values"])

    # Check to make sure all values are valid (ints)
    if not pd.api.types.is_numeric_dtype(df["Values"]):
        print(f"Non-numeric values in: {file_path}")
        print(df["Values"].unique())  # Shows what unusual values are present

    # Convert to datetime
    df["Datetime"] = pd.to_datetime(df["Date"], format="%m/%d/%Y") + pd.to_timedelta(df["Hour"], unit="h")
    data.append(df)
    start_time, end_time = df["Datetime"].min(), df["Datetime"].max()
    
    current_time = start_time
    # Loop through morphology date and retrieve ALL hydrometeor classifications
    while current_time <= end_time:

        # Round down to neareast hour to match morphology assignment and find where they are equal
        morph_row = df[df["Datetime"] == current_time.floor("H")]
        if morph_row.empty:
            current_time += timedelta(minutes=5)
            continue  # no morphology assigned for this hour

        morph_value = morph_row["Values"].values[0]
        
        if morph_value == 0:  # 0 = None in your morphology scheme
            current_time += timedelta(minutes=5)
            continue  # skip this radar time step 
        
        radar_file = radarfuncs.find_closest_radar_lvl3_file(radar_data_dir, var, radar, current_time)
        
        if radar_file:
            try:
                radar_object = pyart.io.read_nexrad_level3(os.path.join(radar_data_dir, radar_file))
                hhc = radar_object.fields["radar_echo_classification"]["data"]

                # Define radar class numbers to exclude
                exclude_classes = [10, 20, 140,150]  # BI, GC, UK
                hhc_filtered = hhc[(hhc >= 10) & (~np.isin(hhc, exclude_classes))]

                # Then compute the histogram
                counts, _ = np.histogram(hhc_filtered, bins=boundaries)
                total = counts.sum()

                if total == 0:
                    continue

                normalized = counts / total
                row = {labels[i]: normalized[i] for i in range(len(labels))} # Dictionary where each hydrometeor is matched with corresponding freqency
                row["Datetime"] = current_time
                row["Event"] = date
                row["Morphology"] = morph_labels[morph_value]

                all_summary_data.append(row)

            except Exception as e:
                print(f"Error at {current_time}: {e}")

        current_time += timedelta(minutes=5)

# Finalize dataframe
# Create a DataFrame from the list of summary data dictionaries
df_all_events = pd.DataFrame(all_summary_data)

# Set the 'Datetime' column as the index (so we can easily access date/time properties)
df_all_events.set_index("Datetime", inplace=True)

# Create a new column 'Month' by extracting the month from the datetime index
df_all_events["Month"] = df_all_events.index.month

# Create a new column 'Season' by applying the custom get_season function to the month
df_all_events["Season"] = df_all_events["Month"].apply(get_season)

# Create a new column 'Hour' by extracting the hour from the datetime index
df_all_events["Hour"] = df_all_events.index.hour

# Create a new column "Datetime' by utilizing
# Hydrometeors to Analyze
hydro_cols = ["GR", "WS","BD","HA"]


hourly_avg = df_all_events.groupby("Hour")[hydro_cols].mean() * 100

print("\n Hourly Average (for each hour) (Selected Hydrometeors):")
print(hourly_avg.round(2))

seasonal_avg = df_all_events.groupby("Season")[hydro_cols].mean() * 100

print("\n Average Frequency by Season:")
print(seasonal_avg.round(2))

event_avg = df_all_events.groupby("Event")[hydro_cols].mean() * 100

print("\n Average per Event:")
print(event_avg.round(2))

band_avg = df_all_events.groupby("Morphology")[hydro_cols].mean() * 100

print("\n Average per Morphology:")
print(band_avg.round(2))


# --- Average Frequency Overall ---
average_freq = df_all_events[hydro_cols].mean() * 100

print(" Overall Average Hydrometeor Frequencies (%):")
for label in hydro_cols:
    print(f"{label}: {average_freq[label]:.2f}%")

# Bar chart of average hydrometeor frequency
plt.figure(figsize=(10, 6))
average_freq.sort_values(ascending=False).plot(kind='bar', color='skyblue', edgecolor='black')

plt.ylabel("Average Frequency (%)")
plt.title("Overall Average Hydrometeor Frequencies")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
#plt.show()

# Bar chart for seasonality changes in morphs
morphology_counts = df_all_events.groupby(["Season", "Morphology"]).size().unstack(fill_value=0)

# Order seasons if needed
season_order = ["Early", "Prime", "Late"]
morphology_counts = morphology_counts.reindex(season_order)

# Plot
morphology_counts.plot(kind="bar", stacked=True, figsize=(10, 6), edgecolor="black")

# If we wanted to keep each season relative frequency normalized to 100%
#relative_counts = morphology_counts.divide(morphology_counts.sum(axis=1), axis=0) * 100
plt.title("Lake-Effect Morphology Frequency by Season",fontsize=22,fontweight='bold')
plt.xlabel("Season",fontsize=16,fontweight='bold')
plt.ylabel("Hours",fontsize=16,fontweight='bold')
plt.legend(title="Morphology")
plt.xticks(rotation=0)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)
#plt.show()

# Export to CSV
df_all_events.to_csv(data_savepath + "hydro_with_morphology.csv", index=True)

colors = {
        0: "green",
        1: "orange",
        2: "purple",
        3: "brown",
        }
# Plotting
plt.figure(figsize=(12, 6))
for idx, col in enumerate(hydro_cols):
    plt.plot(hourly_avg.index, hourly_avg[col], marker='o', label=col,color=colors[idx])

plt.title("Hourly Average Frequency (%) of Hydrometeor Types",fontsize=22,fontweight='bold')
plt.xlabel("Hour (UTC)",fontsize=16,fontweight='bold')
plt.ylabel("Average Frequency (%)",fontsize=16,fontweight='bold')
plt.xticks(range(0, 24))
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title="Hydrometeor Type")
plt.tight_layout()
plt.show()

# Radial Plot!

# Ensure hour is from 0 to 23, then repeat the 0th value at the end to close the circle
hours = np.arange(24)
theta = np.deg2rad(hours * 15)  # convert hours to angles (360/24 = 15° per hour)

# Extend one more point to close the circle
theta = np.append(theta, theta[0])

plt.figure(figsize=(10, 8))
ax = plt.subplot(111, polar=True)

# Plot each hydrometeor type
for col in hydro_cols:
    values = hourly_avg[col].values
    values = np.append(values, values[0])  # close the circle
    ax.plot(theta, values, label=col, linewidth=2)

# Customize
ax.set_theta_direction(-1)  # Clockwise
ax.set_theta_offset(np.pi / 2.0)  # Start at midnight at top
ax.set_xticks(np.deg2rad(np.arange(0, 360, 90)))
ax.set_xticklabels(['00Z', '06Z', '12Z', '18Z'])
ax.set_title("Hourly Hydrometeor Frequency (Radial)", va='bottom')
ax.set_rlabel_position(135)
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.05))

plt.tight_layout()
#plt.show()

# -- Statistical Significance --

group_early = df_all_events[df_all_events["Season"] == "Early"]["GR"]
group_prime = df_all_events[df_all_events["Season"] == "Prime"]["GR"]
group_late = df_all_events[df_all_events["Season"] == "Late"]["GR"]

# Plot histogram with KDE (kernel density estimate)
plt.figure(figsize=(10, 8))

sns.histplot(group_late, bins=20, kde=True)
plt.title("Distribution of Graupel Frequency (Late Season)")
plt.xlabel("Normalized Graupel Frequency")
plt.ylabel("Count")


plt.figure(figsize=(10, 8))

sns.histplot(group_early, bins=20, kde=True)
plt.title("Distribution of Graupel Frequency (Early Season)")
plt.xlabel("Normalized Graupel Frequency")
plt.ylabel("Count")


plt.figure(figsize=(10, 8))

sns.histplot(group_prime, bins=20, kde=True)
plt.title("Distribution of Graupel Frequency (Prime Season)")
plt.xlabel("Normalized Graupel Frequency")
plt.ylabel("Count")
plt.show()

stat, pvalue = levene(group_early.values, group_prime.values, group_late.values)

print(f"Levene's test stat: {stat:.4f}, p-value: {pvalue:.4f}")

statistic, pvalue = kruskal(group_early.values, group_late.values,group_prime.values)

# Print results in scientific notation
print(f"P-value (scientific notation): {pvalue:.2e}")
u_stat, p_val = mannwhitneyu(group_early, group_late, alternative='two-sided')
print(f"Mann-Whitney U p = {p_val:.4e}")
if pvalue == 0:
    print("Exact p = 0 (impossible under normal circumstances)")
else:
    print("P-value is extremely small but not exactly 0")
print(f"Kruskal-Wallis H statistic: {statistic:.4f}")
print(f"P-value: {pvalue:.8f}")

posthoc = sp.posthoc_dunn([group_early, group_prime, group_late], p_adjust='bonferroni')
print(posthoc)
