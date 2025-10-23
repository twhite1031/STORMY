'''
The ECMWF Re-Analysis 5th generation (ERA5) is a global, hourly 
reanalysis dataset from 1940 to present at a 0.25 deg resolution or
roughly 31km at the equator. One of the products of ERA5
is single level data, which contains variables at a single level.
Output format can be in netCDF (.nc) 
or GRIB (.grib). We begin by importing necessary 
packages, such as xarray, for data analysis
'''

import xarray as xr
from datetime import datetime
import cartopy.crs as crs
import matplotlib.pyplot as plt
import STORMY
from STORMY import STORMY_downloader

'''
After importing, we must download the ERA5 Single files given a start and end time, as well as
the variables and area we want. Optionally, we can define a path to save the data to.
'''

start_time = datetime(2022, 11, 18, 13, 50)
end_time = datetime(2022, 11, 18, 15,0)
datapath = ''
plotpath = ''
vars = ['2m_temperature', 'total_precipitation']
area = [60, -115, 40, -90] 

downloader = STORMY_downloader(data_root=datapath)
result = downloader.download_ERA5SINGLE(
    variables=vars,
    start_time=start_time,
    end_time=end_time,
    area=area,
)

ERA5_files = result.files
ERA5_file = ERA5_files[0] # Grab the first file to plot

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
Since we plan on plotting 2m temperature on a plan view map, we need to specify a time to plot.
Then we can extract the 2m temperature variable at that time as well as the lat/lon coordinates
for the extent. 
'''

plot_time = datetime(2022, 11, 18, 14, 0)
t2m = ds['t2m'].sel(time=plot_time) - 273.15  # Convert from Kelvin to Celsius
lat = ds["latitude"].values
lon = ds["longitude"].values

'''
Xarray has two main approaches to reading data:
1) Using the built-in capabilities of xarray to automatically parse variables and coordinates
2) Manually accessing raw variable values and coordinate arrays
We start with the first approach and define a figure.
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

t2m.plot(
    ax=ax,
    transform=crs.PlateCarree(),
    cmap="coolwarm",
    cbar_kwargs={"label": "Temperature (°C)"},
)

plt.title(f"Auto Plot ERA5 2m Temperature at {str(t2m.time.values)[:16]}")

'''
Now lets make the same plot, but manually accessing the raw variable values and coordinate arrays.
We start by defining a new figure.
'''

fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=crs.PlateCarree())
ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=crs.PlateCarree())

'''
Next we grab the raw 2m temperature values
'''

t2m = ds["t2m"].sel(time=plot_time).values - 273.15  # 2D (lat, lon)

'''
Same process for adding map features
'''

STORMY.add_cartopy_features(ax)
STORMY.format_gridlines(ax, x_inline=False, y_inline=False, xpadding=20, ypadding=20)

'''
Now we can plot the data using matplotlib's contourf function and our raw data arrays. We also
make a colorbar and add a fitting title. Since we are using raw values we also need to specify 
the time manually in the title.
'''

im = ax.contourf(lon, lat, t2m, levels=20, cmap="coolwarm", transform=crs.PlateCarree())
plt.colorbar(im, ax=ax, orientation="vertical", label="2m Temperature (°C)")
plt.title(f"Manual Plot ERA5 2m Temperature at {plot_time}")

'''
When we show the plot (plt.show()) these images side by side we see that both 
approaches yield extrememly similar results; 
however, the manual (contourf) approach has its own interpolation method which smooths the data.
'''

plt.savefig(plotpath + "ERA5SINGLETUTORIAL.png")
plt.show()
