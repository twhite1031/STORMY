'''
The National Weather Service (NWS) typically launches two soundings
a day (0Z and 12Z) to gather upper air observations to aid in forecasting
and input into numerical weather models. These soundings are stored by 
Iowa State in .csv files. We being by importing necessary packages.
'''
from metpy.units import units
import metpy.calc as mpcalc
from metpy.plots import SkewT
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import STORMY

'''
After importing, we must define which NWS station(s) you would like to use for the soundings and the 
time range (start and end) you'd like to grab. 
'''

stations = ['KBUF']
start_time, end_time = datetime(2022, 11, 18, 23,55), datetime(2022, 11, 19, 00, 30)
savepath = "/data2/white/DATA/MISC/SOUNDINGS/"

SOUNDING_file = STORMY.download_NWS_SOUNDING(start_time=start_time,end_time=end_time,stations=stations,path_out=savepath)

'''
Now we can read the data, defined by the headers on line 1 (index 0)
'''

df = pd.read_csv(SOUNDING_file)

'''
As with many forms of meteorological data, some quality control (QC)
is required to ensure we can correctly read the values. Here we will
clean missing values, define columns as numbers (instead of strings), 
and convert string time to a datetime object
'''

df.replace("M", pd.NA, inplace=True) # Replace M with nan for easier processing

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='ignore')

if 'validUTC' in df.columns:
    df['validUTC'] = pd.to_datetime(df['validUTC'], errors='coerce')

start_time = df["validUTC"][0]
formatted_time =  start_time.strftime("%Y%m%d_%H%M")

'''
A caveat when downloading multiple stations (and times) of sounding data
is that it is all stored in one file. Therefore we should explicitly define
the start time and station of the sounding we want.
'''
target_time = datetime(2022,11,19,00) # Typically 0Z, 6Z, 12Z, 18Z
station = 'KBUF'

df = df[(df['station'] == station) & (df['validUTC'] == target_time)]

'''
We can now extract the necessary values to create
a sounding. We will drop any nans we added, as well as transfer
the arrays to numpy (for faster/easier processing)
'''

temps = df["tmpc"].dropna().to_numpy() * units.degC
print("Temperatures (°C):", temps)

pressures = df["pressure_mb"].dropna().to_numpy() * units.hPa
print("Pressures (mb):", pressures)

'''
Wind data will need some special attention, since there are nans on some
of the pressure levels. We also need to extract u and v components
for wind barbs.
'''

wind_df = df.dropna(subset=["pressure_mb", "speed_kts", "drct"])
wind_df = wind_df[wind_df["pressure_mb"] >= 100]  # Cap at 100 hPa

wind_pressures = wind_df["pressure_mb"].dropna().to_numpy() * units.hPa
wspd = wind_df["speed_kts"].dropna().to_numpy() * units.knots
wdir = wind_df["drct"].dropna().to_numpy() * units.deg
u_v = mpcalc.wind_components(wspd, wdir) 
u = u_v[0]
v = u_v[1]

'''
A secondary check is done to ensure the temperature and dewpoint temperatures
match when plotting, otherwise the plot will throw an error.
'''
sounding_valid = df[df["tmpc"].notna() & df["dwpc"].notna()] 

p = sounding_valid["pressure_mb"].to_numpy()
T = sounding_valid["tmpc"].to_numpy()
Td = sounding_valid["dwpc"].to_numpy()

'''
Lets use the data we just gathered and QC'd to create a skewT using MetPy's built
in function.
'''
skew = SkewT()

'''
Simply plotting temperature (T) and dewpoint temperature (Td) with
pressure, red and green color respectively
'''
skew.plot(p, T, 'r')
skew.plot(p, Td, 'g')

'''
Plotting wind barbs using our defined valid levels. You can 
index this to adjust spacing (e.g. u[::20])
'''

skew.plot_barbs(wind_pressures, u, v)

'''
Add the iconic skewT lines
'''
skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines()

'''
Set pressure and temperature limits and their labels. Also add a title.
'''

skew.ax.set_ylim(1000, 100)
skew.ax.set_xlim(-40, 20)
skew.ax.set_xlabel('Temperature ($^\circ$C)')
skew.ax.set_ylabel('Pressure (hPa)')
plt.title(f"{station} at {start_time} ")

'''
The skewT is now complete!! Lets create a suitable filename that we can use to save the skewT and
use it in the future. The skewT will be saved using savepath, which you defined earlier.
'''

filename = f"NWS_SOUNDINGTUTORIAL_{formatted_time}.png" # Just to the minute

plt.savefig(savepath + filename)
plt.show()




