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
# ---- User input for file ----

# Format in YEAR, MONTH, DAY, HOUR, MINUTE, DOMAIN
start_time, end_time  = datetime(2023,10,14,12,37,00), datetime(2023, 10, 14, 15, 13, 00)

# Number of time indexes in the wrfout file
# Adjust lufft data freq to match simulationsi (i.e 60 'seconds' for domain 3)
#Domain to look at
domain = 3

if(domain == 3):
    numtimeidx = 20
    new_lufft_freq = 60
elif(domain == 2):
    numtimeidx = 4
    new_lufft_freq = 300
else:
    numtimeidx = 1
    new_lufft_freq = 1200

# Resample or match times for data
resample_lufft = True

# Boolean to save figure
savefig = True

# Lattitude and Longitude of the desired area 
lat_lon = [34.98, -106.04]

# Variable to plot in WRF format
var= 'wspd'

# ---- End User input ----

# ---  Format and open the lufft data file ---

#Column names of Lufft Variables
col_names = ['time','record', 'voltage','ptemperature', 'temperature','rh', 'solar_rad','pressure','wind_speed','wind_dir']

#Reading in the data file
df = pd.read_csv("/data2/white/DATA/NEBP/ANNUL_ECLIP/lufftdata/CR300Series_2_LufftData_all_data_20231019.dat",header=4,delimiter=",",names=col_names)

#Assigning variables to each column of data
temperature = df['temperature'].values
rh = df['rh'].values
solar_rad = df['solar_rad'].values
pressure = df['pressure'].values
wind_speed = df['wind_speed'].values
wind_dir = df['wind_dir'].values
time = df['time'].values
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

def adjust_time_to_resolution(time, timeidx):
    """
    Adjusts a datetime64 time to the nearest valid time based on the resolution.
    """
    minutes = time.astype('datetime64[m]').astype(int) % 60
    if timeidx == 20:
        return time
    elif timeidx  == 4:
        adjusted_minutes = (minutes // 5) * 5
    elif timeidx == 1:
        adjusted_minutes = (minutes // 20) * 20
    else:
        raise ValueError("Unsupported resolution")
    print("Minute value was adjusted to: ", adjusted_minutes)
    return time.astype('datetime64[h]') + np.timedelta64(adjusted_minutes, 'm')

def resample_observational_data(time_array, speed_array, new_freq_sec,start_time=None, end_time=None, timeidx=1):

    # Convert time_array to datetime64 if it's not already
    time_array = np.array(time_array, dtype='datetime64[s]')
    print(time_array) 
    # Adjust start and end times based on resolution
    if start_time is not None:
        start_time = adjust_time_to_resolution(np.datetime64(start_time), timeidx)
    if end_time is not None:
        end_time = adjust_time_to_resolution(np.datetime64(end_time), timeidx)
    
    # Filter data within the specified time range
    if start_time is not None:
        mask_start = time_array >= start_time
    else:
        mask_start = np.ones_like(time_array, dtype=bool)

    if end_time is not None:
        mask_end = time_array <= end_time
    else:
        mask_end = np.ones_like(time_array, dtype=bool)

    # Combine both masks
    mask = mask_start & mask_end
    
    # Apply mask to both time_array and speed_array
    time_array = time_array[mask]
    speed_array = speed_array[mask]
    
    if len(time_array) == 0:
        return np.array([]), np.array([])  # Return empty arrays if no data points remain
    # Convert time_array to seconds since the start
    rounded_time_seconds = (time_array - time_array[0]).astype('timedelta64[s]').astype(int)

    # Round timestamps to the nearest minute
    #rounded_time_seconds = np.round(time_seconds / 60) * 60

    # Convert rounded time back to datetime64
    #rounded_time = time_array[0] + np.array(rounded_time_seconds, dtype='timedelta64[s]')



    # Convert time_array to seconds since the start
    time_seconds = (time_array - time_array[0]).astype('timedelta64[s]').astype(int)

    # Determine the start and end times in seconds
    start_time = time_seconds[0]
    end_time = time_seconds[-1]

    # A string for saving figure (used later)
    resampled = ""
    print("Total number of seconds: ", end_time)
    # Generate new time array with the desired frequency in seconds
    resampled_time_seconds = np.arange(start_time, end_time+.01, new_freq_sec)
    print("Last value in resampled time array", resampled_time_seconds[-1])
    # Resample the speed array
    resampled_speed = []
    if(resample_lufft):
        print("You have chosen to resample the lufft data")
        for t in resampled_time_seconds:
        
        # Find the indices of the original time points that fall within the current resampling interval
            indices = (rounded_time_seconds >= t) & (rounded_time_seconds < t + new_freq_sec)
            if np.any(indices):
                resampled_speed.append(speed_array[indices].mean())
            else:
                resampled_speed.append(np.nan)  # Handle gaps in data
    else:
        print("You have chosen to not resample the lufft data")
        for t in resampled_time_seconds:
            index =int(t/2)
            resampled_speed.append(speed_array[index])

        
    # Convert resampled time in seconds back to datetime
    resampled_time = time_array[0] + np.array(resampled_time_seconds, dtype='timedelta64[s]')
    print("Back to dateimte: ", resampled_time[-1])
    return resampled_time, np.array(resampled_speed)

lufft_1min = resample_observational_data(time, lufftvar, new_lufft_freq, start_time, end_time,numtimeidx)
times, lufft_data = lufft_1min
print('Length of Lufft Data after reshape: ' + str(len(lufft_data)))


# Sort our desired files in order, so our data is also ordered correctly with time
path = f"/data2/white/WRF_OUTPUTS/NEBP/ECLIP_RUN_1/"
path2 = f"/data2/white/WRF_OUTPUTS/NEBP/NON_ECLIP_RUN_1/"

# Retrieve a string from datetime object
wrfstarthour, wrfendhour = int(start_time.strftime("%H")), int(end_time.strftime("%H"))
wrfstartday, wrfendday = int(start_time.strftime("%d")), int(end_time.strftime("%d"))
filelist = []
filelist2 = []
if wrfstarthour < wrfendhour:
    hours = list(range(wrfstarthour, wrfendhour + 1))
    day = np.full(len(list(range(wrfstarthour,wrfendhour+1))), wrfstartday)
else:
    hours = list(range(wrfstarthour, 24)) + list(range(0, wrfendhour+1))
    dayone = np.full(len(list(range(wrfstarthour,24))), wrfstartday)
    daytwo = np.full(len(list(range(0, wrfendhour+1))), wrfendday)
    day = np.append(dayone, daytwo)
for idx, hour in enumerate(hours):
    pattern = path + f'wrfout_d0{domain}_2023-10-{day[idx]}_{hour}'
    filelist += glob.glob(pattern+'*')
    pattern2 = path2 + f'wrfout_d0{domain}_2023-10-{day[idx]}_{hour}'
    filelist2 += glob.glob(pattern2+'*')

# Check the height of our 0 index

with Dataset(filelist[0]) as wrfin:
    x_y = wrf.ll_to_xy(wrfin, lat_lon[0], lat_lon[1])
    phb = wrf.getvar(wrfin,"PHB",timeidx=0)[1,x_y[1],x_y[0]]
    ph =  wrf.getvar(wrfin,"PH",timeidx=0)[1,x_y[1],x_y[0]]
    full_geopt = (phb + ph) / 9.81
    ter_height = wrf.getvar(wrfin,"ter", timeidx=0)[x_y[1],x_y[0]]
    calc_surface_height = full_geopt - ter_height
    print("Calculated height at surface: " , to_np(calc_surface_height))
    height_agl = wrf.getvar(wrfin, "height_agl" , timeidx=0)[0,x_y[1],x_y[0]]
    print("Height agl: ", to_np(height_agl))

def process_file(filename):
    file_data = []
    for i in range(numtimeidx):
        with Dataset(filename) as wrfin:
        # Check if i is double digits for proper time values for x axis
            if(filename[67].isdigit()):
                if(i < 10 and filename[67:69]):
                    time = filename[64:68] + str(i)
                else:
                    time = filename[64:67] + str(int(filename[67:68])+1) + str(i)[1]
            else:
                time = ""
            # Convert desired coorindates to WRF gridbox coordinates
            x_y = wrf.ll_to_xy(wrfin, lat_lon[0], lat_lon[1])

            # Retrieve variables, then focus on one specific gridbox based on our lattitude and longitude input
            #if(var == 'wspd'):
            #    u1 = wrf.getvar(wrfin,"ua",timeidx=i)[0,x_y[1],x_y[0]]
            #    v1 = wrf.getvar(wrfin,"va",timeidx=i)[0,x_y[1],x_y[0]]
            #    # Calculate the wind using the u and v components
            #    datavalue = mpcalc.wind_speed(u1,v1)
            #else:
            datavalue = wrf.getvar(wrfin, var, timeidx=i)
            
            if(to_np(datavalue).ndim == 2):
                datavalue = datavalue[x_y[1],x_y[0]]
            else:
                datavalue = datavalue[0,x_y[1],x_y[0]]
            #
            # Testing this to interpolate variables at 2 meters (or correct heights)
                #value = wrf.getvar(wrfin, var, timeidx=i)
                #height_agl = wrf.getvar(wrfin,"height_agl",timeidx=i)
                #datavalue = wrf.interplevel(value, height_agl, 26.0)[x_y[1],x_y[0]]
                #print(to_np(datavalue))

            # Plot 2d terrain height
            # Plot 


        # Append the obtained values into our empty list
            file_data.append((time, float(datavalue)))
    return file_data

# Parallelize processing of files
with concurrent.futures.ProcessPoolExecutor() as executor:  # Use ProcessPoolExecutor for CPU-bound tasks
    results = executor.map(process_file, sorted(filelist))

# Initialize lists to store results
wrftimes = []
plottingdata = []
plottingdata2 = []

# Unpack and collect the results
for file_data in results:
    for time, datavalue in file_data:
        wrftimes.append(time)
        plottingdata.append(datavalue)

# Convert to numpy arrays
wrftimes = np.array(wrftimes)
times = np.array(times)
plottingdata = np.array(plottingdata)

print("First dataset completed")

# Parallelize processing of files
with concurrent.futures.ProcessPoolExecutor() as executor:  # Use ProcessPoolExecutor for CPU-bound tasks
    results2 = executor.map(process_file, sorted(filelist2))

# Unpack and collect the results
for file_data in results2:
    for time, datavalue in file_data:
        plottingdata2.append(datavalue)
print("Second dataset completed")

# Convert second array to numpy array
plottingdata2 = np.array(plottingdata2)

# Convert units
if(var == "T2"):
    plottingdata = to_np(plottingdata) * units('kelvin')
    plottingdata2 = to_np(plottingdata2) * units('kelvin')
    plottingdata = plottingdata.to('celsius')
    plottingdata2 = plottingdata2.to('celsius')
    plottingdata = plottingdata.magnitude
    plottingdata2 = plottingdata2.magnitude

# To match the lufft

# Function to calculate start and end indices
def calculate_indices(start_time, end_time, timeidx, array):
    start_min = int(start_time.strftime("%M"))
    end_min = int(end_time.strftime("%M"))

    if timeidx == 20:
        start_idx = start_min
        end_idx = -(60 - end_min) + 1
    elif timeidx == 4:
        start_idx = (start_min // 5) 
        end_idx = -((60 - end_min) // 5)  
        if(end_idx == 0):
            end_idx = len(array)
    elif timeidx == 1:
        start_idx = (start_min // 20) 
        print("end_min: ", end_min)
        end_idx = -(((60 - end_min) // 20))  
        if(end_idx == 0):
            end_idx = len(array) 
    else:
        raise ValueError("Unsupported resolution")

    return start_idx, end_idx
start_idx, end_idx = calculate_indices(start_time, end_time, numtimeidx,plottingdata)
print("WRF start idx: ", start_idx)
print("WRF end idx: ", end_idx)
plottingdata = plottingdata[start_idx:end_idx]
plottingdata2 = plottingdata2[start_idx:end_idx]
print("plottingdata: ", plottingdata)
print("plottingdata2: ", plottingdata2)
wrftimes = wrftimes[start_idx:end_idx]
print("Length of adjusted data to match lufft: ", len(plottingdata))
print("First WRF time value (Only accurate if using domain 3): ", wrftimes[0])
print("Last WRF time value (Only accurate if using domain 3): ", wrftimes[-1])
print("First lufft time value: ", times[0])
print("Last lufft time value: ", times[-1])
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

ioa_val_eclip = ioa(plottingdata, lufft_data)
ioa_val_noeclip = ioa(plottingdata2, lufft_data)
ioa_val_sim = ioa(plottingdata, plottingdata2)

# --- Calculate RMSE ----
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

rmse_val_eclip = rmse(plottingdata, lufft_data)
rmse_val_noeclip = rmse(plottingdata2, lufft_data)

# ---- Calculate Systematic RMSE (RMSEs) ----
def rmses(predictions, targets):
    model = LinearRegression().fit(targets.reshape(-1,1),predictions)
    intercept = model.intercept_
    slope = model.coef_[0]
    prediction_star = np.array([])
    prediction_star = intercept + (slope * targets)
    return np.sqrt(((prediction_star - targets) ** 2).mean())

rmses_val_eclip = rmses(plottingdata, lufft_data)
rmses_val_noeclip = rmses(plottingdata2, lufft_data)

# --- Calculate un-systematic RMSE ----
def rmseu(predictions, targets):
    model = LinearRegression().fit(targets.reshape(-1,1),predictions)
    intercept = model.intercept_
    slope = model.coef_[0]
    prediction_star = np.array([])
    prediction_star = intercept + (slope * targets)
    return np.sqrt(((predictions - prediction_star) ** 2).mean())

rmseu_val_eclip = rmseu(plottingdata, lufft_data)
rmseu_val_noeclip = rmseu(plottingdata2, lufft_data)

# --- Calculate mean bias ---
def meanbias(predictions, targets):
    return ((predictions - targets).mean())

meanbias_val_eclip = meanbias(plottingdata, lufft_data)
meanbias_val_noeclip = meanbias(plottingdata2, lufft_data)

# --- Calculate fractional bias ---
def fracbias(predictions, targets):
    return (2 * (predictions.mean() - targets.mean()) / ((predictions.mean() + targets.mean())))

fracbias_val_eclip = fracbias(plottingdata, lufft_data)
fracbias_val_noeclip = fracbias(plottingdata2, lufft_data)

#Create table
data = [
        ['Index of Agreement', ioa_val_eclip, ioa_val_noeclip, ioa_val_sim], \
        ['RMSE', rmse_val_eclip, rmse_val_noeclip ], \
        ['RMSEs', rmses_val_eclip, rmses_val_noeclip], \
        ['RMSEu' , rmseu_val_eclip, rmseu_val_noeclip], \
        ['Mean Bias', meanbias_val_eclip, meanbias_val_noeclip], \
        ['Fractional Bias', fracbias_val_eclip, fracbias_val_noeclip]]

# Make a table
table = tabulate(data,headers = ['Statistic','Lufft vs. WRF Eclipse', 'Lufft vs. WRF No Eclipse', 'WRF Eclipse vs WRF No Eclipse'])
print(table)


# Create a DataFrame
df = pd.DataFrame(data)

# Customize the export settings
custom_header = ['Statistic','Lufft vs. WRF Eclipse', 'Lufft vs. WRF No Eclipse', 'WRF Eclipse vs WRF No Eclipse']

print('times array: ', times)
#df.to_excel('output.xlsx', index=True, na_rep='N/A', header=custom_header, index_label='Statistic')
date_fc = datetime.strptime('2023-10-14T15:13:00', '%Y-%m-%dT%H:%M:%S')
date_max = datetime.strptime('2023-10-14T16:37:00', '%Y-%m-%dT%H:%M:%S')
date_end = datetime.strptime('2023-10-14T18:10:00', '%Y-%m-%dT%H:%M:%S')


# ---- Plotting Sections ----

# Create Figure
fig = plt.figure(figsize=(15, 12))

# Plot data, assign color of line and width
plt.plot(times, plottingdata,color='k',linewidth=2,label='WRF Eclipse')
plt.plot(times, plottingdata2, color='blue',linewidth=2,label='WRF No Eclipse')
plt.plot(times, lufft_data, color='red', linewidth= 2,label='Lufft Observation')

# Here we can adjust the sizing and spacing of the ticks
plt.yticks(fontsize=22)
time_difference = end_time - start_time
min_difference = time_difference.total_seconds() / new_lufft_freq 
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
#plt.ylim(ymin=0)

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

