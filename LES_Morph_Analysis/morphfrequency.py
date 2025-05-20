import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""
Reading a csv file seperate by "," with four columns to identify the Lake-effect band type
and plotting a bargraph of the data
"""

# === USER INPUTS ===
date = "20220123"

# --- End User Input
file_path = f"morphs{date}.csv"
df = pd.read_csv(file_path,sep=r"\s+", header=None, names=["Date", "Hour", "Normal"])

# Convert 'Date' and 'Hour' into a single datetime column
df["Datetime"] = pd.to_datetime(df["Date"], format="%m/%d/%Y") + pd.to_timedelta(df["Hour"], unit="h")

# Define Lake-Effect Band Categories
band_labels = {
    0: "No Lake-Effect",
    1: "Broad Coverage",
    2: "LLAP",
    3: "Hybrid",
    4: "Orographic",
     }

# Count occurrences of each band type for both runs
normal_counts = df["Normal"].value_counts().sort_index()
print("Normal counts: ", normal_counts)

# Define bar width and positions
bar_width = 0.4
norm_indices = np.arange(len(normal_counts))

# Create bar chart
plt.figure(figsize=(8, 5))
plt.bar(norm_indices, normal_counts, width=bar_width, color="skyblue", edgecolor="black")

# Labels and title
plt.xlabel("Lake-Effect Band Type",fontsize=16,fontweight='bold')
plt.ylabel("Hours", fontsize=16,fontweight='bold')
plt.title("Frequency of Lake-Effect Band Types",fontsize=22,fontweight='bold')

# Convert numerical indices to meaningful labels
band_type_labels = [band_labels.get(band, f"Type {band}") for band in normal_counts.index]

# Apply labels
plt.xticks(norm_indices, band_type_labels, rotation=45)

# Add legend
plt.legend()
# Show the plot
plt.show()



