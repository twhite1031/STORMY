import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as mpcalc
from datetime import datetime, timedelta
import pandas as pd

'''
Constructs an area average sounding every hour of an event using ERA5 data in netCDF format
'''

# === User Input ===
date = "20230319" #YYYYMMDD
savepath = f"/data2/white/PLOTS_FIGURES/MET399/{date}/"

# Select region (lat/lon) to average values, currently Lake Ontario Estimate
lat_range = slice(44.15, 43.15)
lon_range = slice(-77.0, -76)
savepath = f"/data2/white/PLOTS_FIGURES/MET399/{date}/"

# === End User Input ===

ds = xr.open_dataset(f"/data2/white/DATA/MET399/ERA5/ERA5{date}.nc")
pressure_levels = ds['pressure_level'].values # Selects all pressure levels (can be adjusted)
start_time = ds["valid_time"][0].values
end_time = ds["valid_time"][-1].values
savepath = f"/data2/white/PLOTS_FIGURES/MET399/{date}/"

# Check units of each variable if applicable
for var in ['t', 'u', 'v', 'q', 'w']:
    if var in ds:
        print(f"{var} units: {ds[var].attrs.get('units', 'No units found')}")
    
# Create subset to apply to dataset
region_subset = ds.sel(
    pressure_level=pressure_levels,
    latitude=lat_range,
    longitude=lon_range,
)


current_time = start_time
current_time = pd.to_datetime(start_time).to_pydatetime()
end_time = pd.to_datetime(end_time).to_pydatetime()
while current_time <= end_time:

    time_subset = region_subset.sel(valid_time=current_time)
    # === Area-average ===
    t = time_subset['t'].mean(dim=['latitude', 'longitude']) * units.kelvin
    q = time_subset['q'].mean(dim=['latitude', 'longitude']) * units('kg/kg')
    u = time_subset['u'].mean(dim=['latitude', 'longitude']) * units('m/s')
    v = time_subset['v'].mean(dim=['latitude', 'longitude']) * units('m/s')
    p = time_subset['pressure_level'].values * units.hPa

    # === Convert variables ===
    Td = mpcalc.dewpoint_from_specific_humidity(p, t, q.metpy.convert_units('g/kg').values * units('g/kg'))
    u = u.values 
    v = v.values 
    T = t.metpy.convert_units('degC').values

    # === Plot on Skew-T ===
    fig = plt.figure(figsize=(8, 10))
    skew = SkewT(fig)

    # Plot data
    skew.plot(p, T, 'r', label='Temp')
    skew.plot(p, Td, 'g', label='Dewpoint')
    skew.plot_barbs(p, u, v)


    # Plot Constants
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines()

    skew.ax.set_ylim(1000, 700)
    skew.ax.set_xlim(-40, 20)
    skew.ax.set_title(f"Area-Averaged Sounding – ERA5 {current_time.strftime('%Y%m%d_%H%M')}", fontsize=14)
    skew.ax.legend()
    
    plt.savefig(savepath + f"ERA5SOUND{current_time.strftime('%Y%m%d_%H%M')}.png")
    plt.close()
    #plt.show()




    current_time += timedelta(hours=1)

