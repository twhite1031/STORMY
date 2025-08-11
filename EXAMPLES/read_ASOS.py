'''
The Automated Surface Observing Systems (ASOS) are a network of weather stations 
operated primarily by NOAA’s National Weather Service, the Federal Aviation Administration, 
and the Department of Defense. These stations are equipped with a suite of sensors to measure key 
surface weather parameters (e.g., temperature, wind speed and direction, precipitation, and visibility). 
This tutorial will work with text-based METAR observation files (.csv) obtained from the ASOS network.
 We begin by importing the necessary packages.
'''

from datetime import datetime
import matplotlib.pyplot as plt
from metpy.units import units
import metpy.calc as mpcalc
from metpy.plots import StationPlot, sky_cover
import cartopy.crs as crs
import numpy as np
import pandas as pd
import STORMY

'''
After importing, we must download the ASOS files given states and a start and end time.
Optionally, we can define a path to save the data to.
'''

states = ["NY"]
start_time, end_time = datetime(2022,11,18,19,50), datetime(2022,11,18,20,10)
ASOS_path = '/data2/white/DATA/MISC/ASOS/'
savepath = '/data2/white/DATA/MISC/ASOS/'

ASOS_file = STORMY.download_ASOS_STATES(
    states=["NY"],
    start_time=start_time,
    end_time=end_time,
    path_out=ASOS_path
    )

'''
Now we can read the data, defined by the headers on line 1 (index 0)
'''

df = pd.read_csv(ASOS_file)

'''
As with many forms of meteorological data, some quality control (QC)
is required to ensure we can correctly read the values. Here we will
clean missing values, define columns as numbers (instead of strings), 
and convert string time to a datetime object
'''

df.replace("M", pd.NA, inplace=True) # Replace M with nan for easier processing

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='ignore')

if 'valid' in df.columns:
    df['valid'] = pd.to_datetime(df['valid'], errors='coerce') 

'''
Some further QC'ing is necessary, ensuring that the rows of data we need are fully
complete.
'''

required_vars = ["tmpf", "dwpf", "mslp", "sknt", "drct", "skyc1", "lat", "lon", "station"]
df = df[df[required_vars].notna().all(axis=1)] # Only keep rows that are complete of required variables

'''
Since we are downloading a range of times, it's best to isolate a particular
time or a small time range for plotting purposes and grab those values. Here we grab 
the latest times for each station, but there's many ways to do this.
'''

df = df.sort_values('valid', ascending=False)
latest_obs = df.groupby('station').first().reset_index()

'''
Now lets read in the values as numpy arrays, applying Metpy units
'''

temps = df["tmpf"].to_numpy() * units.degF
dewpoints = df["tmpf"].to_numpy() * units.degF
pressures = df["mslp"].to_numpy() * units.hPa
wind_speed = df['sknt'].to_numpy() * units.mph
wind_direction = df['drct'].to_numpy() * units.degrees
skyc1 = df['skyc1'].to_numpy()
latitude = df['lat'].to_numpy()
longitude = df['lon'].to_numpy()
station_id = df['station'].to_numpy()

'''
The StationPlot function we are using requires our cloud cover to
be mapped to an representative integer, so we apply that here
'''

sky_map = {
    'CLR': 0, 'SKC': 0, 'NSC': 0, 'NCD': 0, 'CAVOK': 0,  # clear variants
    'FEW': 2,
    'SCT': 4,
    'BKN': 7,
    'OVC': 8,
    'VV ': 8   # vertical visibility, treat as overcast
}

def map_skycover(arr):
    return np.array([sky_map.get(code, np.nan) for code in arr])

skyc1 = map_skycover(skyc1)

'''
Lets create our figure now and apply our projection 
'''

fig = plt.figure(figsize=(16,12))
ax = plt.axes(projection=crs.PlateCarree())

'''
To make a quality figure, we utilize shapefiles that define the borders of lakes, states, countries, 
counties, and more! There are many ways to do this, but I prefer using a helpfer function which
utilizes cartopy.cfeature. Additionally, gridlines are also important to put location and distance 
into perspective, so we will add those as well.
'''

STORMY.add_cartopy_features(ax)
STORMY.format_gridlines(ax, x_inline=False, y_inline=False, xpadding=20, ypadding=20)

'''

'''

stationplot = StationPlot(ax, longitude, latitude, transform=crs.PlateCarree(), fontsize=8)

stationplot.plot_text((2, 0), station_id, fontsize=6)    # ID to right of station
stationplot.plot_parameter('NW', temps, color='red')  # Temp in red
stationplot.plot_parameter('SW', dewpoints, color='green')# Dewpoint in green
u_v = mpcalc.wind_components(wind_speed, wind_direction) 
u = u_v[0]
v = u_v[1]
stationplot.plot_barb(u, v)  

stationplot.plot_parameter('NE', pressures,
                           formatter=lambda v: format(10 * v, '.0f')[-3:])

stationplot.plot_symbol('C', skyc1, sky_cover) # Will fail if any nans 

'''
Nearly complete figure! We now simply add the area we would like to view, title. 
The "f" allows use to put variables in strings, such as the time of the radar scan.
'''

extent = [-80, -72, 40, 45] # Min lon, max lon, min lat, max lat
ax.set_extent(extent, crs=crs.PlateCarree())  
ax.set_title(f"Latest ASOS observations from {start_time} - {end_time}")

'''
The figure is now complete!! Lets create a suitable filename that we can use to save the figure and
use it in the future. The figure will be saved using savepath, which you defined earlier.
'''

start_str, end_str = start_time.strftime('%Y%m%d_%H%M'), end_time.strftime('%Y%m%d_%H%M')
filename = f"ASOSTUTORIAL_{start_str}_{end_str}.png"
plt.savefig(savepath + filename)
plt.show()
