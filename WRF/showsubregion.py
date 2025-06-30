import numpy as np
import matplotlib.pyplot as plt
from wrf import (to_np, getvar, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords)
import cartopy.crs as crs
import cartopy.feature as cfeature
from netCDF4 import Dataset
from datetime import datetime
import cartopy.io.shapereader as shpreader
from cartopy.feature import ShapelyFeature
from shapely.geometry import Polygon
import wrffuncs
import pandas as pd
"""
A plot of a box given the bounds using the same display as your WRF domain, useful if working
with subregions to see what area your focusing on
"""

# --- USER INPUT ---

wrf_date_time = datetime(1997,1,12,1,52,00)
domain = 2
# Bottom left corner , Top right corner for box
lat_lon = [(44.25, -76.25), (43.25,-74.25)]

SIMULATION = "NORMAL" # If comparing runs
path = f"/data2/white/WRF_OUTPUTS/SEMINAR/{SIMULATION}_ATTEMPT/"

# --- END USER INPUT ---

time_df = wrffuncs.build_time_df(path, domain)
obs_time = pd.to_datetime(wrf_date_time)

# Compute absolute time difference
closest_idx = (time_df["time"] - obs_time).abs().argmin()

# Extract the matched row
match = time_df.iloc[closest_idx]

# Unpack matched file info
matched_file = match["filename"]
matched_timeidx = match["timeidx"]
matched_time = match["time"]

print(f"Closest match: {matched_time} in file {matched_file} at time index {matched_timeidx}")


def generate_frame(wrffile, timeidx):
    try:
    # Read data from file
        #print(wrffile)
        with Dataset(wrffile) as ds:
            mdbz = getvar(ds, "ter", timeidx=timeidx)

    # Create a figure
        fig = plt.figure(figsize=(30,15),facecolor='white')

    # Get the latitude and longitude points
        lats, lons = latlon_coords(mdbz)

    # Get the cartopy mapping object
        cart_proj = get_cartopy(mdbz)
    
    # Set the GeoAxes to the projection used by WRF
        ax = plt.axes(projection=cart_proj)
    
    # Special stuff for counties
        reader = shpreader.Reader('/data2/white/PYTHON_SCRIPTS/SEMINAR/countyline_files/countyl010g.shp')
        counties = list(reader.geometries())
        COUNTIES = cfeature.ShapelyFeature(counties, crs.PlateCarree(),zorder=5)
        ax.add_feature(COUNTIES,facecolor='none', edgecolor='black',linewidth=1)

    # Set the map bounds
        ax.set_xlim(cartopy_xlim(mdbz))
        ax.set_ylim(cartopy_ylim(mdbz))
        
        elev_contour = ax.contourf(to_np(lons), to_np(lats), mdbz,levels=np.arange(0, np.max(mdbz), 50), cmap="Greys_r", transform=crs.PlateCarree())
        
        # Add a colorbar
        cbar = plt.colorbar(elev_contour, ax=ax, orientation='vertical', shrink=0.7, pad=0.02)
        cbar.set_label("Terrain Elevation (m)", fontsize=16)
    # Add the gridlines
        gl = ax.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=False)
        gl.xlabel_style = {'rotation': 'horizontal','size': 22,'ha':'center'} # Change 14 to your desired font size
        gl.ylabel_style = {'size': 22}  # Change 14 to your desired font size
        gl.xlines = True
        gl.ylines = True
        gl.top_labels = False  # Disable top labels
        gl.right_labels = False  # Disable right labels
        gl.xpadding = 20
    
        lat1, lon1 = lat_lon[0][0], lat_lon[0][1]  # Bottom-left corner
        lat2, lon2 = lat_lon[1][0], lat_lon[1][1] # Top-right corner
        # Define the coordinates of the square (in order)
        coordinates = [(lon1, lat1), (lon2, lat1), (lon2, lat2), (lon1, lat2), (lon1, lat1)]
        polygon = Polygon(coordinates
                )
        # Create a feature for the polygon
        square_feature = ShapelyFeature([polygon], crs.PlateCarree(), edgecolor='red', facecolor='none',linewidth=3)
        locations = {
            "KTYX Radar": {
                "coords": (43.755, -75.68),
                "color": "red",
                "marker": "^"
            }
        }
        
        #for name, info in locations.items():
        #   lat, lon = info["coords"]
        #    ax.plot(lon, lat, marker=info["marker"], color=info["color"], markersize=10, transform=crs.PlateCarree(), zorder=10)
        #    ax.text(lon + 0.05, lat + 0.05, name, fontsize=16, weight='bold', transform=crs.PlateCarree(), zorder=10,
        #    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
                

        # Add the square to the plot
        ax.add_feature(square_feature)
        plt.show()

    except IndexError:
        print("Error occured")
if __name__ == "__main__":
    generate_frame(matched_file, matched_timeidx)

  
