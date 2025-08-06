import os
from collections import deque
import pandas as pd
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import metpy.calc as mpcalc
from metpy.plots import SkewT
from metpy.units import units
import wrf
from wrf import to_np
import glob
import concurrent.futures
from sklearn.linear_model import LinearRegression
from tabulate import tabulate
from datetime import datetime, timedelta
import xarray as xr
import matplotlib.dates as mdates
import STORMY

# --- USER INPUT ---
start_time, end_time  = datetime(1997,1,10,00,2,00), datetime(1997, 1, 10,00, 18, 00)
domain = 3

# Resample (Average)each interval or match exact times for data
resample_lufft = True
savefig = True # Boolean to save figure

# Lattitude and Longitude of the comparison 
lat_lon = [34.98, -106.04]

var= 'wspd' # WRF variable to plot
path = f"/data2/white/WRF_OUTPUTS/NEBP/ECLIP_RUN_1/"
path2 = f"/data2/white/WRF_OUTPUTS/NEBP/NON_ECLIP_RUN_1/"
lufftpath = "/data2/white/DATA/NEBP/ANNUL_ECLIP/lufftdata/CR300Series_2_LufftData_all_data_20231019.dat"
savepath = f""
# --- END USER INPUT ---

# Build/Find the time data for the model runs
time_df_ECLIP= STORMY.build_time_df(path, domain)
time_df_NOECLIP = STORMY.build_time_df(path, domain)

# Filter time range
mask_ECLIP = (time_df_ECLIP["time"] >= start_time) & (time_df_ECLIP["time"] <= end_time)
mask_NOECLIP = (time_df_NOECLIP["time"] >= start_time) & (time_df_NOECLIP["time"] <= end_time)

time_df_ECLIP = time_df_ECLIP[mask_ECLIP].reset_index(drop=True)
time_df_NOECLIP = time_df_NOECLIP[mask_NOECLIP].reset_index(drop=True)

filelist_ECLIP = time_df_ECLIP["filename"].tolist()
filelist_NOECLIP = time_df_NOECLIP["filename"].tolist()
timelist = time_df_ECLIP["time"].tolist()
timeidxlist = time_df_ECLIP["timeidx"].tolist()

# Adjust lufft frequency based on domain
if(domain == 3):
    new_lufft_freq = 60
elif(domain == 2):
    new_lufft_freq = 300
else:
    new_lufft_freq = 1200

#Column names of Lufft Variables
col_names = ['time','record', 'voltage','ptemperature', 'temperature','rh', 'solar_rad','pressure','wind_speed','wind_dir']

#Reading in the data file
df = pd.read_csv(lufftpath,header=4,delimiter=",",names=col_names)

#Assigning variables to each column of data
temperature = df['temperature'].values
rh = df['rh'].values
solar_rad = df['solar_rad'].values
pressure = df['pressure'].values
wind_speed = df['wind_speed'].values
wind_dir = df['wind_dir'].values
lufft_time = df['time'].values

# Match the WRF variable to the lufft variable
if(var == 'wspd'):
    lufftvar = wind_speed
elif(var == 'T2' or var == 'tc'):
    lufftvar = temperature
elif(var == "SWDNTC"):
    lufftvar = solar_rad
elif(var == "wdir"):
    lufftvar = wind_dir
elif(var == "td2"):
    lufftvar = mpcalc.dewpoint_from_relative_humidity(temperature * units.degC, rh * units.percent)
    lufftvar = lufftvar.magnitude
elif(var == "rh2"):
    lufftvar = rh

def resample_observational_data(time_array, speed_array, target_times, resample_lufft=True):
    """
    Resample observational data to match a list of target datetime64 values.

    Parameters
    ----------
    time_array : array-like of datetime64 or datetime
        Original observation timestamps (e.g., Lufft, every 2s).
    speed_array : array-like of floats
        Observation values.
    target_times : array-like of datetime64
        Target datetimes to resample to (e.g., WRF timesteps).
    resample_lufft : bool
        If True, average values over window around each target time;
        If False, pick value closest to each target time (using t/2 indexing).

    Returns
    -------
    resampled_time : np.ndarray of datetime64[s]
    resampled_speed : np.ndarray of float
    """
    time_array = np.array(time_array, dtype='datetime64[s]')
    speed_array = np.array(speed_array)
    target_times = np.array(target_times, dtype='datetime64[s]')

    if len(time_array) == 0 or len(speed_array) == 0:
        return np.array([]), np.array([])

    # Convert all times to seconds since start of obs for fast indexing
    time_seconds = (time_array - time_array[0]).astype('timedelta64[s]').astype(int)
    target_seconds = (target_times - time_array[0]).astype('timedelta64[s]').astype(int)

    resampled_speed = []

    for i, t_sec in enumerate(target_seconds):
        if resample_lufft:
            # Average over the interval [t, t + model_interval)
            if i < len(target_seconds) - 1:
                next_t_sec = target_seconds[i + 1]
            else:
                # For final time step, define window as +model interval
                model_interval = target_seconds[1] - target_seconds[0]
                next_t_sec = t_sec + model_interval

            indices = (time_seconds >= t_sec) & (time_seconds < next_t_sec)
            resampled_speed.append(speed_array[indices].mean() if np.any(indices) else np.nan)

        else:
            # Lufft is sampled every 2s → grab value at t/2
            idx = int(t_sec / 2)
            if idx < len(speed_array):
                resampled_speed.append(speed_array[idx])
            else:
                resampled_speed.append(np.nan)

    return target_times, np.array(resampled_speed)

# Match WRF times to lufft_times and grab the desired lufft data
times, lufft_data = resample_observational_data(time_array=lufft_time,speed_array=lufft_var,target_times=timelist,resample_lufft=True)  # or False if just picking points)
print('Length of Lufft Data after reshape: ' + str(len(lufft_data)))

def process_file(args):
    filename, timeidx, time = args
    file_data = []
    
    # Convert desired coordinates to WRF gridbox coordinates so we can grab values
    with Dataset(filename) as wrfin:
        x_y = wrf.ll_to_xy(wrfin, lat_lon[0], lat_lon[1])

        '''
        # Retrieve variables, then focus on one specific gridbox based on our lattitude and longitude input
        #if(var == 'wspd'):
        #    u1 = wrf.getvar(wrfin,"ua",timeidx=i)[0,x_y[1],x_y[0]]
        #    v1 = wrf.getvar(wrfin,"va",timeidx=i)[0,x_y[1],x_y[0]]
        #    # Calculate the wind using the u and v components
        #    datavalue = mpcalc.wind_speed(u1,v1)
        '''
        datavalue = wrf.getvar(wrfin, var, timeidx=timeidx)
        
    # If variable is 2D we can grab just x,y otherwise get lowest height level
    if(to_np(datavalue).ndim == 2):
        datavalue = datavalue[x_y[1],x_y[0]]
    else:
        datavalue = datavalue[0,x_y[1],x_y[0]]


    # Append the obtained values into our empty list
        file_data.append((time, float(datavalue)))
    return file_data

# Parallelize processing of files
with concurrent.futures.ProcessPoolExecutor() as executor:  # Use ProcessPoolExecutor for CPU-bound tasks
    tasks = zip(filelist_ECLIP, timeidxlist,timelist)
    results_ECLIP = executor.map(process_file, tasks)

# Initialize lists to store results
wrftimes = []
WRF_ECLIP_DATA = []
WRF_NOECLIP_DATA = []

# Unpack and collect the results
for file_data in results_ECLIP:
    for time, datavalue in file_data:
        wrftimes.append(time)
        WRF_ECLIP_DATA.append(datavalue)
print("First dataset completed")

# Convert to numpy arrays
wrftimes = np.array(wrftimes)
times = np.array(times)
WRF_ECLIP_DATA = np.array(WRF_ECLIP_DATA)

# Parallelize processing of files
with concurrent.futures.ProcessPoolExecutor() as executor:  # Use ProcessPoolExecutor for CPU-bound tasks
    tasks = zip(filelist_NOECLIP, timeidxlist,timelist)
    results_NOECLIP = executor.map(process_file, tasks)

# Unpack and collect the results
for file_data in results_NOECLIP:
    for time, datavalue in file_data:
        WRF_NOECLIP_DATA.append(datavalue)
print("Second dataset completed")

# Convert second array to numpy array
WRF_NOECLIP_DATA = np.array(WRF_NOECLIP_DATA)

# Convert units if needed to match lufft data
if(var == "T2"):
    WRF_ECLIP_DATA = to_np(WRF_ECLIP_DATA) * units('kelvin')
    WRF_NOECLIP_DATA = to_np(WRF_NOECLIP_DATA) * units('kelvin')
    WRF_ECLIP_DATA = WRF_ECLIP_DATA.to('celsius')
    WRF_NOECLIP_DATA = WRF_NOECLIP_DATA.to('celsius')
    WRF_ECLIP_DATA = WRF_ECLIP_DATA.magnitude
    WRF_NOECLIP_DATA = WRF_NOECLIP_DATA.magnitude

# Sanity checks
print("First WRF time value: ", wrftimes[0])
print("Last WRF time value: ", wrftimes[-1])
print("First Lufft time value: ", times[0])
print("Last Lufft time value: ", times[-1])
print("Length of WRF data: ", len(WRF_ECLIP_DATA))

# ---- Calculate Index of Agreement -----
def ioa(predictions, targets):
    # Mean of targets
    mean_observed = np.mean(targets)

    # Calculate the numerator and denominator
    numerator = np.sum((predictions - targets) ** 2)
    denominator = np.sum((np.abs(predictions - mean_observed) + np.abs(targets - mean_observed)) ** 2)

    # Index of Agreement
    index_of_agreement = 1 - (numerator / denominator)

    return index_of_agreement

ioa_val_eclip = ioa(WRF_ECLIP_DATA, lufft_data)
ioa_val_noeclip = ioa(WRF_NOECLIP_DATA, lufft_data)
ioa_val_sim = ioa(WRF_ECLIP_DATA, WRF_NOECLIP_DATA)

# --- Calculate RMSE ----
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

rmse_val_eclip = rmse(WRF_ECLIP_DATA, lufft_data)
rmse_val_noeclip = rmse(WRF_NOECLIP_DATA, lufft_data)

# ---- Calculate Systematic RMSE (RMSEs) ----
def rmses(predictions, targets):
    model = LinearRegression().fit(targets.reshape(-1,1),predictions)
    intercept = model.intercept_
    slope = model.coef_[0]
    prediction_star = np.array([])
    prediction_star = intercept + (slope * targets)
    return np.sqrt(((prediction_star - targets) ** 2).mean())

rmses_val_eclip = rmses(WRF_ECLIP_DATA, lufft_data)
rmses_val_noeclip = rmses(WRF_NOECLIP_DATA, lufft_data)

# --- Calculate un-systematic RMSE ----
def rmseu(predictions, targets):
    model = LinearRegression().fit(targets.reshape(-1,1),predictions)
    intercept = model.intercept_
    slope = model.coef_[0]
    prediction_star = np.array([])
    prediction_star = intercept + (slope * targets)
    return np.sqrt(((predictions - prediction_star) ** 2).mean())

rmseu_val_eclip = rmseu(WRF_ECLIP_DATA, lufft_data)
rmseu_val_noeclip = rmseu(WRF_NOECLIP_DATA, lufft_data)

# --- Calculate mean bias ---
def meanbias(predictions, targets):
    return ((predictions - targets).mean())

meanbias_val_eclip = meanbias(WRF_ECLIP_DATA, lufft_data)
meanbias_val_noeclip = meanbias(WRF_NOECLIP_DATA, lufft_data)

# --- Calculate fractional bias ---
def fracbias(predictions, targets):
    return (2 * (predictions.mean() - targets.mean()) / ((predictions.mean() + targets.mean())))

fracbias_val_eclip = fracbias(WRF_ECLIP_DATA, lufft_data)
fracbias_val_noeclip = fracbias(WRF_NOECLIP_DATA, lufft_data)

# Create table structure
data = [
        ['Index of Agreement', ioa_val_eclip, ioa_val_noeclip, ioa_val_sim], \
        ['RMSE', rmse_val_eclip, rmse_val_noeclip ], \
        ['RMSEs', rmses_val_eclip, rmses_val_noeclip], \
        ['RMSEu' , rmseu_val_eclip, rmseu_val_noeclip], \
        ['Mean Bias', meanbias_val_eclip, meanbias_val_noeclip], \
        ['Fractional Bias', fracbias_val_eclip, fracbias_val_noeclip]]

# Create the table
table = tabulate(data,headers = ['Statistic','Lufft vs. WRF Eclipse', 'Lufft vs. WRF No Eclipse', 'WRF Eclipse vs WRF No Eclipse'])
print(table)

# Create a DataFrame
df = pd.DataFrame(data)

# Customize the export settings
custom_header = ['Statistic','Lufft vs. WRF Eclipse', 'Lufft vs. WRF No Eclipse', 'WRF Eclipse vs WRF No Eclipse']

# Set first contact, max, and end of the eclipse
date_fc = datetime(2023,10,14,15,13,0)
date_max = datetime(2023,10,14,16,37,0)
date_end = datetime(2023,10,14,18,10,0)

# ---- Plotting Sections ----

# Create Figure
fig = plt.figure(figsize=(15, 12))

# Plot data, assign color of line and width
plt.plot(times, WRF_ECLIP_DATA,color='k',linewidth=2,label='WRF Eclipse')
plt.plot(times, WRF_NOECLIP_DATA, color='blue',linewidth=2,label='WRF No Eclipse')
plt.plot(times, lufft_data, color='red', linewidth= 2,label='Lufft Observation')

# Here we can adjust the sizing and spacing of the ticks
plt.yticks(fontsize=22)
plt.xticks(fontsize=14,rotation=45)
plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=20))  # major ticks every 10 minutes
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))     # format as hour:minute

# Here we plot the time of annularity in relation to our dataset
plt.axvline(x=date_max,linestyle='--', color='gray')
plt.axvspan(date_fc, date_max, facecolor='peachpuff', alpha=0.5)
plt.axvspan(date_max, date_end, facecolor='gold', alpha=0.5)

# Here we give our axes labels
plt.xlabel('Time (UTC)',fontsize=24,fontweight='bold')
plt.ylabel('m/s',fontsize=18,fontweight='bold')
plt.legend(fontsize=14)

# This line makes the y axis start at the bottom, rather than slightly above the bottom
plt.ylim(ymin=0)

# Format datetime for title and name of figure
fmt_start_time, fmt_end_time = start_time.strftime('%Y-%m-%d %H:%M:%S'), end_time.strftime('%Y-%m-%d %H:%M:%S')

if(resample_lufft):
    resampled = "RS"
else:
    resampled = ""

# Here we assign a fitting title
plt.title(f"Surface Wind Speed (m/s) from {fmt_start_time}Z to {fmt_end_time}Z",fontsize=20,fontweight='bold')

# Save the figure
if savefig:
    plt.savefig(f'/data2/white/PLOTS_FIGURES/NEBP/ECLIP_RUN_1/lufftlineD{domain}{var}{fmt_start_time}{fmt_end_time}{resampled}',dpi=300)

plt.show()

