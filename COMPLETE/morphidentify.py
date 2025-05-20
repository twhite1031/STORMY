import pandas as pd
import matplotlib.pyplot as plt

"""
Reading a csv file seperate by "," with four columns to identify the Lake-effect band type
and plotting how band changes over time using a line (step) plot
"""

# Sample DataFrame (Replace with actual file)
file_path = "morphologydata.csv"
df = pd.read_csv(file_path, sep=",", header=None, names=["Date", "Hour", "Normal", "Flat"])

# Convert 'Date' and 'Hour' into a single datetime column
df["Datetime"] = pd.to_datetime(df["Date"], format="%m/%d/%Y") + pd.to_timedelta(df["Hour"], unit="h")

# Convert non-numeric values properly
#df["Normal"] = pd.to_numeric(df["Run1"], errors="coerce")
#df["Flat"] = pd.to_numeric(df["Run2"], errors="coerce")

# Define Lake-Effect Band Categories
band_labels = {
    0: "No Lake-Effect",
    1: "Broad Coverage",
    2: "LLAP",
    3: "Hybrid",
    4: "Orographic",
    5: "Unknown"
}

# Create the Step Plot
plt.figure(figsize=(12, 5))
plt.step(df["Datetime"], df["Normal"], label="Normal Simulation", where="mid", linestyle="-", color="b")
plt.step(df["Datetime"], df["Flat"], label="Flat Simulation", where="mid", linestyle="--", color="r")

# Apply Custom Y-Tick Labels
plt.yticks(list(band_labels.keys()), list(band_labels.values()))

# Formatting
plt.xlabel("Time", fontsize=16,fontweight='bold')
plt.ylabel("Lake Effect Band Type",fontsize=16,fontweight='bold')
plt.title("Lake Effect Band Evolution Over Time",fontsize=22,fontweight='bold')
plt.xticks(rotation=45)
plt.legend()
plt.grid()

# Show the plot
plt.show()
