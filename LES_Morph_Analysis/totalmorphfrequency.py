import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Your event dates
dates = ["20181110", "20200227", "20220110", "20220123", "20230319","20201102","20221117","20191231","20190305"]
data = []

# Read and combine data from all files
for date in dates:
    file_path = f"/data2/white/DATA/MET399/morphs/morphs{date}.csv"
    df = pd.read_csv(file_path, delim_whitespace=True, header=None, names=["Date", "Hour", "Values"])
    if not pd.api.types.is_numeric_dtype(df["Values"]):
        print(f"Non-numeric values in: {file_path}")
        print(df["Values"].unique())  # Shows what unusual values are present
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

# Count total occurrences of each band type
total_counts = all_data["Values"].value_counts().sort_index()
print("Total band counts across all events:\n", total_counts)

# Calculate total number of observations
total_observations = total_counts.sum()
print("total counts 0", total_counts[0])

# Calculate percentages
percentages = (total_counts / (total_observations - total_counts[0])) * 100

# Print out as a table
print("Frequency and Percentage of Each Band Type:\n")
for band_code in total_counts.index:
    label = band_labels.get(band_code, f"Type {band_code}")
    count = total_counts[band_code]
    pct = percentages[band_code]
    print(f"{label}: {count} occurrences ({pct:.2f}%)")

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

season_counts = all_data.groupby(["Season", "Values"]).size().unstack(fill_value=0)

for season in season_counts.index:
    print(f"\n{season} Season:")
    for morph_code in season_counts.columns:
        label = band_labels.get(morph_code, f"Type {morph_code}")
        count = season_counts.loc[season, morph_code]
        print(f"  {label}: {count} hours")

# Assume 0 = "No Lake-Effect"
les_only = all_data[all_data["Values"] != 0]

# Group by season and count
season_hour_totals = les_only["Season"].value_counts().sort_index()

# Print results
print("Total identifiable LES hours per season:")
for season, count in season_hour_totals.items():
    print(f"{season}: {count} hours")
# Plot
bar_width = 0.4
indices = np.arange(len(total_counts))

plt.figure(figsize=(8, 5))
plt.bar(indices, total_counts, width=bar_width, color="skyblue", edgecolor="black")

plt.xlabel("Lake-Effect Band Type")
plt.ylabel("Hours")
plt.title("Total Hours of Lake-Effect Band Types (All Events)")

# Apply labels to x-axis
band_type_labels = [band_labels.get(band, f"Type {band}") for band in total_counts.index]
plt.xticks(indices, band_type_labels, rotation=45)

plt.tight_layout()
plt.show()









