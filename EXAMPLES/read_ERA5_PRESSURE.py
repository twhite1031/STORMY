'''
The ECMWF Re-Analysis 5th generation (ERA5) is a global, hourly 
reanalysis dataset from 1940 to present at a 0.25 deg resolution or
roughly 31km at the equator. One of the products of ERA5
is pressure level data, which contains variables that are distributed
throughout various pressure levels. Output format can be in netCDF (.nc) 
or GRIB (.grib). We begin by importing necessary 
packages, such as xarray, for data analysis
'''

from metpy.plots import SkewT
import xarray as xr
from datetime import datetime
import cartopy.crs as crs
import matplotlib.pyplot as plt
import STORMY
from STORMY import STORMY_downloader
import numpy as np

'''
After importing, we must download the ERA5 Single files given a start and end time, as well as
the variables and area we want. Optionally, we can define a path to save the data to.
'''

start_time = datetime(2022, 11, 18, 13, 50)
end_time = datetime(2022, 11, 18, 15,0)
plotpath = ''
vars = ["relative_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
    ]
area = [60, -115, 40, -90] 

downloader = STORMY_downloader(data_root='')
result = downloader.download_ERA5PRESSURE(
    variables=vars,
    start_time=start_time,
    end_time=end_time,
    area=area,
)

ERA5_files = result.files
ERA5_file = ERA5_files[0] # Grab the first file to plot
print(f"Using ERA5 file: {ERA5_file}")

'''
Now we can use xarray to read in the grib file we just downloaded that we can read the data in using xarray. 
Note that we specify the engine as "cfgrib" since xarray does not natively read grib files.
'''

ds = xr.open_dataset(ERA5_file, engine="cfgrib")
print(ds) # View the dataset structure

'''
As always, we like to see what variables and coordinates are available in the dataset, this can 
be done by using the data_vars and coords attributes of the xarray.Dataset object.
'''

print(f"Variables in dataset: {list(ds.data_vars)}")
print(f"Coordinates in dataset: {list(ds.coords)}")

'''
Since we plan on plotting m temperature on a plan view map, we need to specify a time to plot as well as a pressure level.
Then we can extract the 2m temperature variable at that time.
'''

plot_time = datetime(2022, 11, 18, 14, 0)
pressure_level = 500
t = ds['t'].sel(time=plot_time,isobaricInhPa=pressure_level) - 273.15  # Convert from Kelvin to Celsius

'''
Xarray has two main approaches to reading data:
1) Using the built-in capabilities of xarray to automatically parse variables and coordinates
2) Manually accessing raw variable values and coordinate arrays
We will only use the first approach here and define a figure. See read_ERA5_SINGLE.py for
an example of both approaches.
'''

fig = plt.figure(figsize=(10,8))
ax = plt.axes(projection=crs.PlateCarree())
ax.set_extent([-115, -90, 40, 60], crs=crs.PlateCarree())

'''
To make a quality figure, we utilize shapefiles that define the borders of lakes, states, countries, 
counties, and more! There are many ways to do this, but I prefer using a helpfer function which
utilizes cartopy.cfeature. Additionally, gridlines are also important to put location and distance 
into perspective, so we will add those as well.
'''

STORMY.add_cartopy_features(ax)
STORMY.format_gridlines(ax, x_inline=False, y_inline=False, xpadding=20, ypadding=20)

'''
This is the automatic plotting approach, where xarray handles the lat/lon coordinates
and plotting for us "under the hood". 
'''

t.plot(
    ax=ax,
    transform=crs.PlateCarree(),
    cmap="coolwarm",
    cbar_kwargs={"label": "Temperature (°C)"},
)

plt.title(f"ERA5 {pressure_level}hPa Temperature at {str(t.time.values)[:16]}")
plt.savefig(plotpath + "ERA5PRESSURETUTORIAL.png")
plt.show()

'''
Since we have vertical levels in this dataset, we can also create a skewT diagram at a specific location.
You can also do an area average or vertical cross section, I encourage you to explore those options as well.
We will choose the University of Grand Forks coordinates for our reanalysis sounding.
'''

lat_UND, lon_UND = 47.9253, -97.0328 
T = ds['t'].sel(time=plot_time,latitude=lat_UND,longitude=lon_UND,method='nearest') - 273.15  # Convert from Kelvin to Celsius
rh = ds['r'].sel(time=plot_time,latitude=lat_UND,longitude=lon_UND,method='nearest')  # Relative humidity in %
u = ds['u'].sel(time=plot_time,latitude=lat_UND,longitude=lon_UND,method='nearest')  # U wind component in m/s
v = ds['v'].sel(time=plot_time,latitude=lat_UND,longitude=lon_UND,method='nearest')  # V wind component in m/s
p = ds['isobaricInhPa'].values  # Pressure levels in hPa

'''
We need to get dewpoint temperature from relative humidity, a quick helper function
will do this for us. We also rewrite Td as an xarray DataArray to keep the metadata.
'''

def dewpoint_from_rh(temp_c, rh_pct):
    a, b = 17.625, 243.04
    rh = np.clip(rh_pct, 1e-6, 100.0)  # avoid log(0)
    alpha = np.log(rh / 100.0) + (a * temp_c) / (b + temp_c)
    dewpoint_c = (b * alpha) / (a - alpha)
    return dewpoint_c

Td = dewpoint_from_rh(T.values, rh.values)
Td = xr.DataArray(Td, coords=T.coords, dims=T.dims, name="Td")
Td.attrs['units'] = 'degC'
Td.attrs['long_name'] = 'Dew point temperature'

'''
Now we can use the new dewpoint temperature along with temperature and wind data to create a skewT diagram.
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

skew.plot_barbs(p, u, v)

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
plt.title(f"University of North Dakota at {start_time} ")

'''
The skewT is now complete!! Lets create a suitable filename that we can use to save the skewT and
use it in the future. The skewT will be saved using savepath, which you defined earlier.
'''

plt.savefig(plotpath + "ERA5PRESSURETUTORIAL_SkewT.png")
plt.show()
