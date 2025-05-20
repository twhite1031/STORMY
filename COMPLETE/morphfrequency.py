import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""
Reading a csv file seperate by "," with four columns to identify the Lake-effect band type
and plotting a bargraph of the data
"""

# Sample DataFrame (Replace with actual file)
file_path = "morphologydata.csv"
df = pd.read_csv(file_path, sep=",", header=None, names=["Date", "Hour", "Normal", "Flat"])

# Convert 'Date' and 'Hour' into a single datetime column
df["Datetime"] = pd.to_datetime(df["Date"], format="%m/%d/%Y") + pd.to_timedelta(df["Hour"], unit="h")

# Define Lake-Effect Band Categories
band_labels = {
    0: "None",
    1: "Broad Coverage",
    2: "LLAP",
    3: "Hybrid",
    4: "Orographic",
}

# Count occurrences of each band type for both runs
normal_counts = df["Normal"].value_counts().sort_index()
print("Normal counts: ", normal_counts)
flat_counts = df["Flat"].value_counts().sort_index()
print("Flat counts: ", flat_counts)
# Define bar width and positions
bar_width = 0.4
norm_indices = np.arange(len(normal_counts))
flat_indices = np.arange(len(flat_counts))

# Create bar chart
plt.figure(figsize=(8, 5))
plt.bar(norm_indices, normal_counts, width=bar_width, color="skyblue", edgecolor="black", label="Normal Simulation")
plt.bar(flat_indices + bar_width, flat_counts, width=bar_width, color="salmon", edgecolor="black", label="Flat Simulation")

# Labels and title
plt.xlabel("Lake-Effect Band Type",fontsize=16,fontweight='bold')
plt.ylabel("Frequency",fontsize=16,fontweight='bold')
plt.title("Frequency of Lake-Effect Band Types by Model Run",fontsize=20,fontweight='bold')

# Convert numerical indices to meaningful labels
band_type_labels = [band_labels.get(band, f"Type {band}") for band in normal_counts.index]

# Apply labels
plt.xticks(norm_indices + bar_width / 2, band_type_labels, rotation=45)

# Add legend
plt.legend()
# Show the plot
plt.show()
