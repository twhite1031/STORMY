import numpy as np
import pandas as pd # http://pandas.pydata.org/
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pyart
import glob
import datetime as dt
import nexradaws
from metpy.plots import USCOUNTIES
import os
import pickle
import requests
import warnings
import h5py

time = dt.datetime(2022,11,19,00,00)
lma_on = True

min_events_per_flash = 10 # Minimum number of sources per flash
min_stations = 6 # more stations = more confident it's a good solution
max_chi = 1 # lower chi^2 = more confident it's a good solution
tbuffer = 15*60 # range for LMA data with specified time in middle in seconds

lma_point_color = 'fuchsia' # LMA point color.
lma_point_size = 25 # LMA point size
lma_point_marker = '^' # LMA point marker.


def get_LMA_flash_data(start):
    if lma_on == True:
        filenames = []
        flashes = []
        flash_event_time = []
        flash_events = []
        selection_event = []
        lma_lon = []
        lma_lat = []
        #os.makedirs('lma', exist_ok=True)
        filename = '/data2/white/DATA/PROJ_LEE/LMADATA/LYLOUT_{}000_0600.dat.flash.h5'.format(start.strftime('%y%m%d_%H%M')[:-1])
        filenames.append(filename)

        if (glob.glob(filename) == []):
            url = 'https://data.nssl.noaa.gov/thredds/fileServer/WRDD/OKLMA/deployments/flashsort_6/h5_files/{}/{}'.format(start.strftime('%Y/%m/%d'),os.path.basename(filename))
            response = requests.get(url)
            with open(filename, "wb") as file:
                file.write(response.content)
            print(f'{filename} downloaded successfully.')

        if (tbuffer > 600):
            for i in range(int(tbuffer/600)):
                filename = '/data2/white/DATA/PROJ_LEE/LMADATA/LYLOUT_{}000_0600.dat.flash.h5'.format((start+dt.timedelta(seconds=(i*600))).strftime('%y%m%d_%H%M')[:-1])
                filenames.append(filename)
                if (glob.glob(filename) == []):
                    url = 'https://data.nssl.noaa.gov/thredds/fileServer/WRDD/OKLMA/deployments/flashsort_6/h5_files/{}/{}'.format((start+dt.timedelta(seconds=(i*600))).strftime('%Y/%m/%d'),os.path.basename(filename))
                    print(url)
                    response = requests.get(url)
                    with open(filename, "wb") as file:
                        file.write(response.content)
                    print(f'{filename} downloaded successfully.')
        filename = '/data2/white/DATA/PROJ_LEE/LMADATA/LYLOUT_{}000_0600.dat.flash.h5'.format((start+dt.timedelta(seconds=tbuffer)).strftime('%y%m%d_%H%M')[:-1])
        if filename not in filenames:
            filenames.append(filename)
            if (glob.glob(filename) == []):
                    url = 'https://data.nssl.noaa.gov/thredds/fileServer/WRDD/OKLMA/deployments/flashsort_6/h5_files/{}/{}'.format((start+dt.timedelta(seconds=tbuffer)).strftime('%Y/%m/%d'),os.path.basename(filename))
                    response = requests.get(url)
                    with open(filename, "wb") as file:
                        file.write(response.content)
                    print(f'{filename} downloaded successfully.')
                    
        flashes = pd.DataFrame()
        flash_events = pd.DataFrame()
        for filename in filenames:
            timeobj = dt.datetime.strptime(filename.split('/')[-1], 
                                           "LYLOUT_%y%m%d_%H%M%S_0600.dat.flash.h5")
            # This is the flash table
            flashes2 = pd.read_hdf(filename,'flashes/LMA_{}00_600'.format(timeobj.strftime('%y%m%d_%H%M')))
            # This is the event (VHF source) table
            flash_events2 = pd.read_hdf(filename,'events/LMA_{}00_600'.format(timeobj.strftime('%y%m%d_%H%M')))
            # Flash ID's are not unique between files. This writes new ones 
            # in the second file, if it exists
            if flashes.shape[0]>0:
                flashes2.flash_id      = flashes2['flash_id']     +flashes.flash_id.max()+1
                flash_events2.flash_id = flash_events2['flash_id']+flashes.flash_id.max()+1
            else:
                pass
            flashes      = pd.concat([flashes,flashes2])
            flash_events = pd.concat([flash_events,flash_events2])         

        # Make a series of datetime objects for each event
        flash_event_time = np.array([dt.datetime(*start.timetuple()[:3])+dt.timedelta(seconds = j) for j in flash_events.time])
        try:
            # Select all the sources meeting the criteria set above
            selection_event = (flash_event_time>=start)&(flash_event_time < start+dt.timedelta(seconds=tbuffer))&(flash_events.chi2<=max_chi)&(flash_events.stations>=min_stations)
            
            lma_lon = (flash_events.lon[selection_event].values)
            
            lma_lat = (flash_events.lat[selection_event].values)

                       
            
        except AttributeError:
            pass

get_LMA_flash_data(time)

# Define the directory and filename
filename = "/data2/white/DATA/PROJ_LEE/LMADATA/LYLOUT_221119_001000_0600.dat.flash.h5"

timeobj = dt.datetime.strptime(filename.split('/')[-1], 
                                           "LYLOUT_%y%m%d_%H%M%S_0600.dat.flash.h5")
# This is the flash table
flashes = pd.read_hdf(filename,'flashes/LMA_{}00_600'.format(
                                            timeobj.strftime('%y%m%d_%H%M')))
# This is the event (VHF source) table
flash_events = pd.read_hdf(filename,'events/LMA_{}00_600'.format(
                                          timeobj.strftime('%y%m%d_%H%M')))
print(flashes)


# Make a series of datetime objects for each event
flash_event_time = np.array([dt.datetime(*time.timetuple()[:3])+dt.timedelta(seconds = j) for j in flash_events.time]) # Retrieve day and calculate event time based on seconds from midnight
print(flash_event_time)
# Select all the sources meeting the criteria set above
selection_event = (flash_event_time>=time)&(flash_event_time < time+dt.timedelta(seconds=tbuffer))&(flash_events.chi2<=max_chi)&(flash_events.stations>=min_stations)
            
lma_lon = (flash_events.lon[selection_event].values)
            
lma_lat = (flash_events.lat[selection_event].values)

print(flashes)
print(flash_events)


